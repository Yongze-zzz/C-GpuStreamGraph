// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#include <framework/framework.cuh>
#include <framework/hybrid_policy.h>
#include <framework/clion_cuda.cuh>
#include <framework/variants/api.cuh>
#include "hybrid_pr_common.h"
#include <functional>
#include <map>

// Priority
DEFINE_double(cut_threshold, 0, "Cut threshold for index calculation");
DEFINE_bool(sparse, false, "disable load-balancing for sparse graph");
DECLARE_double(error);
DECLARE_int32(top_ranks);
DECLARE_bool(print_ranks);
DECLARE_string(output);

namespace hybrid_pr
{
    template<typename TValue, typename TBuffer, typename TWeight, typename...UnusedData>
    struct PageRank : sepgraph::api::AppBase<TValue, TBuffer, TWeight>
    {

        /*
         * For get rid of compiler bug: It's strange that if base class has virtual function, we must add a member for subclass.
         *
         * Error: Internal Compiler Error (codegen): "there was an error in verifying the lgenfe output!"
         */
        double m_error;
        using sepgraph::api::AppBase<TValue, TBuffer, TWeight>::AccumulateBuffer;
        PageRank(double error) : m_error(error)
        {
	      
        }

        __forceinline__ __device__

        TValue GetInitValue(index_t node) const override
        {
            return 0.0f;
        }

        __forceinline__ __device__

        TBuffer GetInitBuffer(index_t node) const override
        {
            return (1 - ALPHA);
        }

        __forceinline__ __host__
        __device__
                TBuffer

        GetIdentityElement() const override
        {
            return 0.0f;
        }

        __forceinline__ __device__

        utils::pair<TBuffer, bool> CombineValueBuffer(index_t node,
                                                      TValue *p_value,
                                                      TBuffer *p_buffer) override
        {
            TBuffer buffer = atomicExch(p_buffer, IDENTITY_ELEMENT);
            // TValue *outv;
            bool schedule = false;
            // if (buffer > 0.01)
            // if (buffer > m_error)
            if (buffer != 0)
            {
                schedule = true;
                *p_value += buffer;
                int out_degree = this->m_vcsr_graph.sync_vertices_[node].degree;
                
                buffer = ALPHA * buffer / out_degree;
                // if (node == 1 || node==2 || node==3 || node ==4)LOG("#node %d, degree %d, value %f, buffer %f\n", node, out_degree, *p_value, buffer);
            }
            return utils::pair<TBuffer, bool>(buffer, schedule);
        }

        __forceinline__ __device__
        utils::pair<TBuffer, bool> CombineValueBufferAmend(index_t node,int *type_device,
                                                      TValue *p_value,
                                                      TBuffer *p_buffer) override
        {
            // TBuffer buffer = atomicExch(p_buffer, IDENTITY_ELEMENT);
            // TBuffer buffer = atomicExch(p_buffer, *p_buffer);
            TBuffer buffer =  (*p_value) * (type_device[0]);
            //  TBuffer buffer =  (*p_value);
            bool schedule = false;

            if (buffer != 0)
            {
                schedule = true;
                // schedule = false;
                int out_degree = this->m_vcsr_graph.sync_vertices_[node].degree;
                buffer = ALPHA * buffer / out_degree;
            }
            return utils::pair<TBuffer, bool>(buffer, schedule);
        }

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
            //p_buffer is dst delta ,buffer is src delta
            // if(dst ==5 )printf("acc src %d -> dst %d delta %f\n",src,dst,buffer);
            atomicAdd(p_buffer, buffer);
            return 0;
        }

        __forceinline__ __device__

        bool IsActiveNode(index_t node, TBuffer buffer, TValue value) const override
        {
            return buffer > m_error;
        }

        __forceinline__ __device__

        bool IsActiveNode_INC(index_t node, TBuffer buffer, TValue value) const override
        {
            return buffer < (-m_error) ;
        }
        
        __forceinline__ __device__

        TValue sum_value(index_t node, TValue value, TBuffer buffer) const override
        {
            return buffer;
        }

        __forceinline__ __device__

        bool IsHighPriority(TBuffer current_priority, TBuffer buffer) const override
        {
            return current_priority <= buffer;
        }
    };
}

bool HybridPageRank()
{
    LOG("HybridPageRank\n");
    typedef sepgraph::engine::Engine<rank_t, rank_t, rank_t, hybrid_pr::PageRank, double> HybridEngine;
    HybridEngine engine(sepgraph::policy::AlgoType::ITERATIVE_SCHEME);
    sepgraph::engine::EngineOptions engine_opt;

    LOG("error %f\n", (float) FLAGS_error);
    engine.SetOptions(engine_opt);
    engine.LoadGraph();
    engine.InitGraph(FLAGS_error);
    ///冷启动
    engine.Start();
    //热度计算(.cache = false .delta = false)
    engine.compute_hot_vertices_pr();
    
    //确定候选顶点(.cache = false .delta = false)
    engine.confirm_candidate_batch();

    //缓存加载(.cache = false hot_node.delta = true)
    engine.LoadCache();
    engine.get_update_file();
    std::pair<index_t,index_t> local_begin;
    local_begin.first = 0; //first is add
    local_begin.second = 0; // second is del
    index_t NumOfSnapShots = 0;
    // while(true){
    //     if(NumOfSnapShots==10) break;

        engine.Cancelation(local_begin,NumOfSnapShots);
        engine.Compensate(local_begin,NumOfSnapShots);
        //打印缓存中的顶点热度
        // engine.PrintCacheNode_2();
        engine.compute_hot_vertices_pr();
        engine.confirm_candidate_batch();
        //将仍然在缓存中的顶点合并，分配新的缓存索引，以及更新缓存目前水位
        engine.compact_cache();
        //重新根据热度计算需要缓存的顶点，首先热度根据多个快照热度排序，然后根据排序好的点集合提取度数，
        //PrefixSum《目前水位的顶点需要重新传输给缓存，提取度数会跳过被缓存的顶点
        // engine.evication();
        engine.LoadCache();
        // // cudaDeviceSynchronize();
        NumOfSnapShots+=1;
        engine.Cancelation(local_begin,NumOfSnapShots);
        engine.Compensate(local_begin,NumOfSnapShots);
        NumOfSnapShots+=1;
        engine.compute_hot_vertices_pr();
        engine.confirm_candidate_batch();
        engine.compact_cache();
        engine.LoadCache();
        // engine.PrintCacheNode_2();
    // }
    // engine.evication();
    // // engine.compute_hot_vertices_pr();
    // // //确定候选顶点(.cache = false .delta = false)
    // // engine.confirm_candidate_batch();

    // 
    engine.GatherValue();
    engine.GatherBuffer();
    const auto &distances = engine.GetGraphDatum().host_value;
    const auto &deltas = engine.GetGraphDatum().host_buffer;
    cudaDeviceSynchronize();
    for(index_t i = 0; i<20; i++){
        printf("v %d data %f delta %f\n",i,distances[i],deltas[i]);
        
    } 

    bool success = true;

        printf("Warning: Result not checked\n");

    return success;
}