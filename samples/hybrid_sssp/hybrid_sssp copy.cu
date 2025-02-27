// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#include <functional>
#include <map>
#include <framework/framework.cuh>
#include <framework/hybrid_policy.h>
#include <framework/clion_cuda.cuh>
#include <utils/cuda_utils.h>
#include "hybrid_sssp_common.h"
#include "../../include/groute/graphs/csr_graph.cuh"

DEFINE_int32(source_node,
             0, "The source node for the SSSP traversal (clamped to [0, nnodes-1])");
DEFINE_bool(sparse,
            false, "use async/push/dd + fusion for high-diameter");
DECLARE_int32(top_ranks);
DECLARE_bool(print_ranks);
DECLARE_string(output);
DECLARE_bool(check);
DECLARE_int32(prio_delta);

namespace hybrid_sssp
{
    template<typename TValue, typename TBuffer, typename TWeight, typename...UnusedData>
    struct SSSP : sepgraph::api::AppBase<TValue, TBuffer, TWeight>
    {
        using sepgraph::api::AppBase<TValue, TBuffer, TWeight>::AccumulateBuffer;
        index_t m_source_node;

        SSSP(index_t source_node) : m_source_node(source_node)
        {
         
        }

        __forceinline__ __device__

        TValue GetInitValue(index_t node) const override
        {
            return static_cast<TValue> (IDENTITY_ELEMENT);
        }

        __forceinline__ __device__

        TBuffer GetInitBuffer(index_t node) const override 
        {
            TBuffer buffer;
            if (node == m_source_node)//source_node = 1
            {
                buffer = 0;
            }
            else
            {
                buffer = IDENTITY_ELEMENT;
            }
            return buffer;
        }

        __forceinline__ __host__ __device__
        TBuffer GetIdentityElement() const override
        {
            return IDENTITY_ELEMENT;
        }

        __forceinline__ __device__
        utils::pair<TBuffer, bool> CombineValueBufferAmend(index_t node,int *type_device,
                                                      TValue *p_value,
                                                      TBuffer *p_buffer) override
        {
           
            return utils::pair<TBuffer, bool>(*p_buffer, true);
        }
        __forceinline__ __device__
        utils::pair<TBuffer, bool> CombineValueBuffer(index_t node,
                                                      TValue *p_value,
                                                      TBuffer *p_buffer) override
        {
            TBuffer buffer = *p_buffer;
            bool schedule = false;
            if (*p_value > buffer)
            {
                *p_value = buffer;
                schedule = true;
            }
            return utils::pair<TBuffer, bool>(buffer, schedule);
        }

        // 3: if new_path==result[dst] then
        // 4: old_p=parent[dst]
        // 5: repeat
        // 6: old_p= atomicCAS(&parent[dst],src)
        // 7: until (new_path!= result[dst]||old_p==parent[dst])
        __forceinline__ __device__
        int AccumulateBuffer(index_t src,
                             index_t dst,
                            //  TValue level, //src_level
                            //  TValue *p_level, //dst_level
                             TWeight weight,
                             TValue *p_parent,
                             TBuffer *p_buffer,    //dst_buffer
                             TBuffer buffer) override   //src_buffer
        {
            // TBuffer old_buffer = *p_buffer;
            TBuffer new_buffer = buffer + weight;
            TBuffer old_buffer=atomicMin(p_buffer, buffer + weight);;
            if(new_buffer< old_buffer){
            
                TValue old_parent = *p_parent;
                do{
                    old_parent = atomicCAS(p_parent,old_parent,src);
                    
                } while (new_buffer==*p_buffer && (*p_parent) !=src);
            }
            return 1;
        }

        __forceinline__ __device__
        int AccumulateBuffer_add(index_t src,
                             index_t dst,
                            //  TValue level, //src_level
                            //  TValue *p_level, //dst_level
                             TWeight weight,
                             TValue *p_parent, //dst_parent
                             TValue *p_buffer,    //dst_value
                             TValue *buffer) override   //src_value
        {
            TValue incoming_value_curr = *buffer;
            if(incoming_value_curr ==UINT32_MAX){
                return 0;
            }
            TValue new_buffer = incoming_value_curr + weight;
            TValue new_parent = src;
            // TValue new_level = level+1;
            TValue old_value;
            old_value = atomicMin(p_buffer, new_buffer);
            TValue old_parent = *p_parent;
        // TValue curr_buffer = *p_buffer;
            // if(new_buffer == curr_buffer){
            do{
                if(new_buffer==old_value){
                    old_parent = atomicCAS(p_parent,old_parent,src);
                }
            } while (new_buffer==old_value && old_parent !=src);
            return 1;
        }

        // __forceinline__ __device__
        // bool reduce()

        __forceinline__ __device__
        int AccumulateBuffer_del(index_t src,
                             index_t dst,
                             TValue *p_parent,
                             TBuffer *p_buffer,    //dst_buffer
                             TValue *p_value) override   //dst_value
        {
            if(*p_parent == src){
                *p_buffer = UINT32_MAX;
                *p_value = UINT32_MAX;
                *p_parent = UINT32_MAX;
                this->m_vcsr_graph.vertices_[dst].deletion = true;
            }
            return 1;
        }

        __forceinline__ __device__

        bool IsActiveNode(index_t node, TBuffer buffer,TValue value) const override
        {
            return buffer < value;
        }
        
        
        __forceinline__ __device__
        TValue sum_value(index_t node, TValue value,TBuffer buffer) const override
        {
            if(value > buffer * 2)
                return TValue(2);

            return TValue(1);
        }


        __forceinline__ __device__

        bool IsHighPriority(TBuffer current_priority, TBuffer buffer) const override
        {
            return current_priority > buffer;
        }
    };
}


/**
 * Δ = cw/d,
    where d is the average degree in the graph, w is the average
    edge weight, and c is the warp width (32 on our GPUs).
 * @return
 */

bool HybridSSSP()
{
    assert(UINT32_MAX == UINT_MAX);
    // typedef sepgraph::engine::Engine<distance_t, distance_t, distance_t, hybrid_sssp::SSSP, index_t> HybridEngine;
    typedef sepgraph::engine::Engine<distance_t, distance_t, distance_t, hybrid_sssp::SSSP, index_t> HybridEngine;
    HybridEngine engine(sepgraph::policy::AlgoType::TRAVERSAL_SCHEME);
    engine.LoadGraph();

    index_t source_node = min(max((index_t) 0, (index_t) FLAGS_source_node), engine.GetGraphDatum().nnodes - 1);

    sepgraph::common::EngineOptions engine_opt;


    groute::graphs::host::PMAGraph vcsr_graph = engine.PMAGraph();
    double weight_sum = 0;

    /**
     * We select a similar heuristic, Δ = cw/d,
        where d is the average degree in the graph, w is the average
        edge weight, and c is the warp width (32 on our GPUs)
        Link: https://people.csail.mith.edu/jshun/papers/DBGO14.pdf
     */
    // int init_prio = 32 * (weight_sum / vcsr_graph.nedges) /
    //                 (1.0 * csr_graph.nedges / csr_graph.nnodes);
    int init_prio = 32 * (weight_sum / vcsr_graph.nedges) /
                (1.0 * vcsr_graph.nedges / vcsr_graph.nnodes);

    printf("Priority delta: %u\n", init_prio);

    if (FLAGS_sparse)
    {
        engine_opt.SetFused();
        engine_opt.SetTwoLevelBasedPriority(init_prio);
        engine_opt.ForceVariant(sepgraph::common::AlgoVariant::ASYNC_PUSH_DD);
        engine_opt.SetLoadBalancing(sepgraph::common::MsgPassing::PUSH, sepgraph::common::LoadBalancing::NONE);
    }

    if (FLAGS_prio_delta > 0)
    {
        LOG("Enable priority for scale-free dataset\n");
        engine_opt.SetTwoLevelBasedPriority(FLAGS_prio_delta);
    }

    engine.SetOptions(engine_opt);
    engine.InitGraph(source_node);
    engine.Start(init_prio);
    //PrintCacheNode() are used to detect the cache is right or not.
    bool success = true;
    engine.compute_hot_vertices_sssp();
    engine.confirm_candidate_batch();
    engine.LoadCache();
    // engine.PrintCacheNode();
    engine.get_update_file();
    std::pair<index_t,index_t> local_begin;
    local_begin.first = 0; //first is add
    local_begin.second = 0; // second is del
    index_t NumOfSnapShots = 0;
    while(true){
        if(NumOfSnapShots==10) break;
        std::cout<<"batch number "<<NumOfSnapShots<<std::endl;
        engine.del_edge(local_begin,NumOfSnapShots);
        engine.add_edge(local_begin,NumOfSnapShots);
        engine.compute_hot_vertices_sssp();
        engine.confirm_candidate_batch();
        engine.evication_cache();
        engine.compact_cache();
        engine.LoadCache();
        NumOfSnapShots+=1;
    }
    engine.GatherValue();
    engine.GatherParent();
    engine.GatherBuffer();
    const auto &distances = engine.GetGraphDatum().host_value;
    const auto &parents = engine.GetGraphDatum().host_parent;
    const auto &deltas = engine.GetGraphDatum().host_buffer;
    // std::ofstream valid("/home/hdd/yongze/datasets/quafu-3.txt",std::ios::out|std::ios::trunc);
    // if(!valid.is_open()){
    //     std::cerr << "Error: Unable to open file for writing." << std::endl;
    //     return 1;
    // }
    // for(index_t i = 0; i<vcsr_graph.nnodes; i++){
    //     uint32_t vd;
    //     if(distances[i]== UINT32_MAX){
    //          vd = INT32_MAX;
    //     }else{
    //         vd = distances[i];
    //     }
    //     // printf("v %d data %d parent %d\n",i,distances[i],parents[i]);
    //     valid<<i<<" "<<vd<<" "<<parents[i]<<std::endl;
    // } 
    
    cudaDeviceSynchronize();
    return success;
}