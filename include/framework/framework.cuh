// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_FRAMEWORK_H
#define HYBRID_FRAMEWORK_H

#include <functional>
#include <map>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thread>
#include <thrust/sort.h>
#include <cub/cub.cuh>
#include <framework/common.h>
#include <framework/variants/api.cuh>
#include <framework/graph_datum.cuh>
#include <framework/variants/common.cuh>
#include <framework/variants/driver.cuh>
#include <framework/hybrid_policy.h>
#include <framework/algo_variants.cuh>
#include <utils/cuda_utils.h>
#include <utils/graphs/traversal.h>
#include <utils/to_json.h>
#include <groute/device/work_source.cuh>
#include "clion_cuda.cuh"
#include <groute/rmat_util.h>
#include "Loader.h"
#include <unordered_set>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

DECLARE_int32(residence);
DECLARE_int32(priority_a);
DECLARE_int32(hybrid);
DECLARE_int32(SEGMENT);
DECLARE_int32(n_stream);
DECLARE_int32(max_iteration);
DECLARE_string(out_wl);
DECLARE_string(lb_push);
DECLARE_string(lb_pull);
DECLARE_double(alpha);
DECLARE_bool(undirected);
DECLARE_bool(wl_sort);
DECLARE_bool(wl_unique);

DECLARE_double(edge_factor);
DECLARE_string(updatefile);
DECLARE_string(update_size);
DECLARE_bool(weight);

namespace sepgraph {
    namespace engine {
        using common::Priority;
        using common::LoadBalancing;
        using common::Scheduling;
        using common::Model;
        using common::MsgPassing;
        using common::AlgoVariant;
        using policy::AlgoType;
        using policy::PolicyDecisionMaker;
        using utils::JsonWriter;


        struct Algo {
            static const char *Name() {
                return "Hybrid Graph Engine";
            }
        };  



      template<typename TValue, typename TBuffer, typename TWeight, template<typename, typename, typename, typename ...> class TAppImpl, typename... UnusedData>
      class 	Engine {
        private:
        typedef TAppImpl<TValue, TBuffer, TWeight, UnusedData...> AppImplDeviceObject;
        typedef graphs::GraphDatum<TValue, TBuffer, TWeight> GraphDatum;
        typedef Loader<index_t, index_t, index_t> Loader;
        // typedef EdgePair<uint32_t,uint32_t> Edge;
        typedef WEdgePair<uint32_t,uint32_t,uint32_t>  WEdge;
        // using WeightedDynT = groute::graphs::single::TestGraph<TValue, TBuffer, TValue,true>;
        cudaDeviceProp m_dev_props;
	    
            // Graph data
            std::unique_ptr<utils::traversal::Context<Algo>> m_groute_context;
            std::unique_ptr<groute::Stream> m_stream;
            std::unique_ptr<groute::graphs::single::CSRGraphAllocator> m_csr_dev_graph_allocator;
            std::unique_ptr<groute::graphs::single::PMAGraphAllocator> m_vcsr_dev_graph_allocator;
            // std::unique_ptr<groute::graphs::single::PMAGraphAllocator> m_vcsr_dev_graph_allocator_update;
            std::unique_ptr<groute::graphs::single::CSCGraphAllocator> m_csc_dev_graph_allocator;

            std::unique_ptr<AppImplDeviceObject> m_app_inst;
            groute::Stream stream[64];
            // App instance
            std::unique_ptr<GraphDatum> m_graph_datum;
            std::unique_ptr<Loader> m_load_update;
            policy::TRunningInfo m_running_info;
            TBuffer current_priority;  // put it into running info
            PolicyDecisionMaker m_policy_decision_maker;
            EngineOptions m_engine_options;
            WEdge* added_edges_h=nullptr;
            WEdge* added_edges_d=nullptr;
            WEdge* del_edges_h=nullptr;
            WEdge* del_edges_d=nullptr;
            uint32_t* work_size_d = nullptr;
            int type[1];
            int *type_device=nullptr;
            //partition_information
            unsigned int partitions_csc;
            unsigned int* partition_offset_csc;
            unsigned int max_partition_size_csc;

            unsigned int partitions_csr;
            unsigned int* partition_offset_csr;
            unsigned int max_partition_size_csr;
            // Loader<index_t,index_t,index_t> load_update;
            // WeightedDynT result_graph;

            public:
            Engine(AlgoType algo_type) :
            m_running_info(algo_type),
            m_policy_decision_maker(m_running_info) {
                int dev_id = 0;

                GROUTE_CUDA_CHECK(cudaGetDeviceProperties(&m_dev_props, dev_id));
                m_groute_context = std::unique_ptr<utils::traversal::Context<Algo>>
                (new utils::traversal::Context<Algo>(1));

		        //create stream /*CODE by ax range 118 to 121*/
                for(int i = 0; i < FLAGS_n_stream; i++)
                {
                    stream[i] = m_groute_context->CreateStream(dev_id);
                }
		
        m_stream = std::unique_ptr<groute::Stream>(new groute::Stream(dev_id));
    }

    void SetOptions(EngineOptions &engine_options) {
        m_engine_options = engine_options;
    }

    index_t GetNodeNum(){
      return m_groute_context->host_graph.nnodes;
    }
            void compute_hot_vertices_pr(){
                GraphDatum &graph_datum = *m_graph_datum;
                auto &app_inst = *m_app_inst;
                groute::Stream &stream_s = *m_stream;
                const auto &work_source = groute::dev::WorkSourceRange<index_t>(0, graph_datum.nnodes); 
                const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                const auto &hvcsr = m_vcsr_dev_graph_allocator->HostObject();
                dim3 grid_dims, block_dims;
                KernelSizing(grid_dims, block_dims, work_source.get_size());
                Stopwatch sw_ch(true);
                kernel::comp_hotness_pr<< < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (app_inst,
                    vcsr_graph,
                    work_source,
                    graph_datum.m_node_buffer_datum,
                    graph_datum.d_hotness);
                stream_s.Sync();
                sw_ch.stop();
                LOG("comp_hotness time: %f ms (excluded)\n", sw_ch.ms());
                graph_datum.sort_vtx_by_hotness();
                // graph_datum.CompareDeviceResult();
                Stopwatch extrac(true);
                kernel::extract_vtx_degree<< < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (app_inst,
                    vcsr_graph,
                    work_source,
                    graph_datum.d_id,
                    graph_datum.d_v);
                stream_s.Sync();
                extrac.stop();
                kernel::reset_hotness<< < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (app_inst,
                    vcsr_graph,
                    work_source,
                    graph_datum.d_hotness);
                    stream_s.Sync();
                LOG("extract degree time: %f ms (excluded)\n", extrac.ms());
                cudaDeviceSynchronize();
            }

            void compute_hot_vertices_sssp(){
                GraphDatum &graph_datum = *m_graph_datum;
                auto &app_inst = *m_app_inst;
                groute::Stream &stream_s = *m_stream;
                const auto &work_source = groute::dev::WorkSourceRange<index_t>(0, graph_datum.nnodes); 
                const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                const auto &hvcsr = m_vcsr_dev_graph_allocator->HostObject();
                dim3 grid_dims, block_dims;
                KernelSizing(grid_dims, block_dims, work_source.get_size());
                Stopwatch sw_ch(true);
                kernel::comp_hotness_sssp<< < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (app_inst,
                    vcsr_graph,
                    work_source,
                    graph_datum.m_node_buffer_datum,
                    graph_datum.d_hotness);
                stream_s.Sync();
                sw_ch.stop();
                LOG("comp_hotness time: %f ms (excluded)\n", sw_ch.ms());
                graph_datum.sort_vtx_by_hotness();
                LOG("sort vtx ms (excluded)\n");
                cudaDeviceSynchronize();
                // graph_datum.CompareDeviceResult();
                Stopwatch extrac(true);
                kernel::extract_vtx_degree<< < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (app_inst,
                    vcsr_graph,
                    work_source,
                    graph_datum.d_id,
                    graph_datum.d_v);
                stream_s.Sync();
                extrac.stop();
                kernel::reset_hotness<< < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (app_inst,
                    vcsr_graph,
                    work_source,
                    graph_datum.d_hotness);
                    stream_s.Sync();
                LOG("extract degree time: %f ms (excluded)\n", extrac.ms());
                cudaDeviceSynchronize();
            }

            void confirm_candidate_batch(){
                GraphDatum &graph_datum = *m_graph_datum;
                auto &app_inst = *m_app_inst;
                groute::Stream &stream_s = *m_stream;
                Stopwatch extrac(true);
                graph_datum.ensure_candidate_vertex();
                extrac.stop();
                LOG("candidate v time: %f ms (excluded)\n", extrac.ms());
                const auto &work_source = groute::dev::WorkSourceRange<index_t>(0, graph_datum.nnodes); 
                const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                // const auto &hvcsr = m_vcsr_dev_graph_allocator->HostObject();
                dim3 grid_dims, block_dims;
                KernelSizing(grid_dims, block_dims, work_source.get_size());
                     Stopwatch search_rebuild(true);   
                kernel::search_batch<< < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (app_inst,
                vcsr_graph,
                work_source,
                graph_datum.num_of_cache_d,
                graph_datum.d_sum,
                graph_datum.d_id.Current());
                stream_s.Sync();
                search_rebuild.stop();
                LOG("search_rebuild v time: %f ms (excluded)\n", search_rebuild.ms());
                cudaDeviceSynchronize();         
            }

            void LoadCache(){
                auto &app_inst = *m_app_inst;
                GraphDatum &graph_datum = *m_graph_datum;
                Loader &load_update = *m_load_update;
                groute::Stream &stream_s = *m_stream;
                auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                AlgoVariant next_policy[FLAGS_SEGMENT];
                for(index_t i = 0; i < FLAGS_SEGMENT; i++){
                    next_policy[i] = m_policy_decision_maker.GetInitPolicy();
                }
                Stopwatch sw_load(true); 
                index_t seg_snode,seg_enode;
                index_t stream_id;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
                    seg_snode = m_groute_context->seg_snode[seg_idx];
                    seg_enode = m_groute_context->seg_enode[seg_idx];  
                        RebuildWorklist_delta(app_inst,vcsr_graph,
                        graph_datum,
                        stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                }
                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                    stream[stream_idx].Sync();
                }
                ExecutePolicy_MF_topo(next_policy);
                cudaDeviceSynchronize();
                sw_load.stop();
                LOG("加载缓存数据时间  : %f ms \n",sw_load.ms());
            }

            void ExecutePolicy_MF_topo(AlgoVariant *algo_variant) {
                auto &app_inst = *m_app_inst;
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
		        index_t seg_snode,seg_enode;
                uint64_t seg_sedge_csr,seg_nedges_csr;
                index_t stream_id;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
    		        seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
    		        seg_enode = m_groute_context->seg_enode[seg_idx];                                    // end node
    		        stream_id = seg_idx % FLAGS_n_stream;
                    const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                    if(algo_variant[seg_idx] == AlgoVariant::Zero_Copy){
                        m_vcsr_dev_graph_allocator->SwitchZC();
                        zcflag = true;
                        RunSyncPushDDBAmend_cache(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx,zcflag,
                           vcsr_graph,
                        //    type_device,
                           graph_datum,
                           m_engine_options,
                           stream[stream_id]); 
                    }
               }
               for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }
            }
//Loadgraph from the data structure to the segment_structure
        void LoadGraph() {
      Stopwatch sw_load(true);
      groute::graphs::host::PMAGraph &vcsr_graph = m_groute_context->host_pma_small;
      index_t m_nsegs = FLAGS_SEGMENT;
      uint64_t seg_sedge_csr, seg_eedge_csr;
      index_t seg_snode,seg_enode, seg_nnodes;
      uint64_t seg_nedges_csr;
      uint64_t seg_nedges = round_up(vcsr_graph.elem_capacity, m_nsegs);
	  uint64_t seg_nedges_csr_max = 0;  //dev memory		    
	  uint64_t edge_num = 0;		    
	  index_t node_id = 0;
	  uint64_t out_degree;
	  std::vector<index_t> nnodes_num;
	  seg_snode = node_id;
	  m_groute_context->seg_snode[0] = seg_snode;
      vcsr_graph.sync_vertices_[vcsr_graph.nnodes].index =  vcsr_graph.elem_capacity;
      for(index_t seg_idx = 0; node_id < vcsr_graph.nnodes ; seg_idx++){
          m_groute_context->seg_snode[seg_idx] = node_id;
          while(edge_num < seg_nedges){
            out_degree = vcsr_graph.end_edge(node_id) - vcsr_graph.begin_edge(node_id);
            edge_num = edge_num + out_degree;
            if(node_id < vcsr_graph.nnodes){
                node_id ++;
            }else{
                break;
            }
           }
           if(node_id == vcsr_graph.nnodes){
                seg_enode = node_id ; 
            }
           else{
                seg_enode = node_id;	    
           }
            seg_nnodes = seg_enode - seg_snode;

            m_running_info.nnodes_seg[seg_idx] = seg_nnodes;
            nnodes_num.push_back(seg_nnodes);
            
            m_groute_context->seg_enode[seg_idx] = seg_enode;	
            seg_sedge_csr = vcsr_graph.sync_vertices_[seg_snode].index;
            seg_eedge_csr = vcsr_graph.sync_vertices_[seg_enode].index;                
            seg_nedges_csr = seg_eedge_csr - seg_sedge_csr;

            m_running_info.total_workload_seg[seg_idx] = seg_nedges_csr;
            seg_nedges_csr_max = max(seg_nedges_csr_max,seg_nedges_csr);

            m_groute_context->seg_sedge_csr[seg_idx] = seg_sedge_csr; 
            m_groute_context->seg_nedge_csr[seg_idx] = seg_nedges_csr;

            // LOG("seg_idx : %d, seg_snode : %d,seg_enode : %d, seg_sedge_csr : %d,seg_eedge_csr : %d, seg_nedge_csr : %d\n", seg_idx, seg_snode, seg_enode, m_groute_context->seg_sedge_csr[seg_idx], seg_eedge_csr, m_groute_context->seg_nedge_csr[seg_idx]);
            edge_num = 0;
            seg_snode = node_id;		     
        }

        m_groute_context->segment_ct = FLAGS_SEGMENT;
        m_groute_context->SetDevice(0);

        // m_csr_dev_graph_allocator = std::unique_ptr<groute::graphs::single::CSRGraphAllocator>(
        //     new groute::graphs::single::CSRGraphAllocator(csr_graph,seg_nedges_csr_max));
        LOG("seg_nedges_csr_max = %d\n",seg_nedges_csr_max);
        m_vcsr_dev_graph_allocator = std::unique_ptr<groute::graphs::single::PMAGraphAllocator>(new groute::graphs::single::PMAGraphAllocator(vcsr_graph,seg_nedges_csr_max));

        m_graph_datum = std::unique_ptr<GraphDatum>(new GraphDatum(vcsr_graph,seg_nedges_csr_max,nnodes_num));

        sw_load.stop();

        m_running_info.time_load_graph = sw_load.ms();

        LOG("Load graph time: %f ms (excluded)\n", sw_load.ms());

        m_running_info.nnodes = m_groute_context->nvtxs;
        m_running_info.nedges = m_groute_context->nedges;
        m_running_info.total_workload = m_groute_context->nedges * FLAGS_edge_factor;
        current_priority = m_engine_options.GetPriorityThreshold();
    }
    /*
        * Init Graph Value and buffer fields
        */
    
        void InitGraph(UnusedData &...data) {
            Stopwatch sw_init(true);
            
            m_app_inst = std::unique_ptr<AppImplDeviceObject>(new AppImplDeviceObject(data...));
            groute::Stream &stream_s = *m_stream;
            GraphDatum &graph_datum = *m_graph_datum;
            const auto &dev_vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
            const auto &work_source = groute::dev::WorkSourceRange<index_t>(0, graph_datum.nnodes); 
            dim3 grid_dims, block_dims;

            m_app_inst->m_vcsr_graph = dev_vcsr_graph;
            m_app_inst->m_nnodes = graph_datum.nnodes;
            m_app_inst->m_nedges = graph_datum.nedges;
            m_app_inst->m_p_current_round = graph_datum.m_current_round.dev_ptr;
            // Launch kernel to init value/buffer fields
            KernelSizing(grid_dims, block_dims, work_source.get_size());
            auto &app_inst = *m_app_inst;
            // auto &result_dyn_graph = result_graph.dyn();
            // result_dyn_graph.Allocate(graph_datum.nnodes);
            kernel::InitGraph
            << < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (app_inst,
                work_source,
                graph_datum.d_id,
                graph_datum.d_hotness,
                graph_datum.GetParentDeviceObject(),
                graph_datum.GetValueDeviceObject(),
                graph_datum.GetBufferDeviceObject());
            stream_s.Sync();

            index_t seg_snode,seg_enode;
            index_t stream_id;

            for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                stream_id = seg_idx % FLAGS_n_stream;
                seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
                seg_enode = m_groute_context->seg_enode[seg_idx];  
                    RebuildArrayWorklist(app_inst,
                    graph_datum,
                    stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
            }

            for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                stream[stream_idx].Sync();
            }
            // m_groute_context->host_pma_small.PrintHistogram(graph_datum.m_in_degree.dev_ptr,graph_datum.m_out_degree.dev_ptr);
            m_running_info.time_init_graph = sw_init.ms();

            sw_init.stop();

            LOG("InitGraph: %f ms (excluded)\n", sw_init.ms());	
        }

        void SaveToJson() {
            JsonWriter &writer = JsonWriter::getInst();

            writer.write("time_input_active_node", m_running_info.time_overhead_input_active_node);
            writer.write("time_output_active_node", m_running_info.time_overhead_output_active_node);
            writer.write("time_input_workload", m_running_info.time_overhead_input_workload);
            writer.write("time_output_workload", m_running_info.time_overhead_output_workload);
            writer.write("time_queue2bitmap", m_running_info.time_overhead_queue2bitmap);
            writer.write("time_bitmap2queue", m_running_info.time_overhead_bitmap2queue);
            writer.write("time_rebuild_worklist", m_running_info.time_overhead_rebuild_worklist);
            writer.write("time_priority_sample", m_running_info.time_overhead_sample);
            writer.write("time_sort_worklist", m_running_info.time_overhead_wl_sort);
            writer.write("time_unique_worklist", m_running_info.time_overhead_wl_unique);
            writer.write("time_kernel", m_running_info.time_kernel);
            writer.write("time_total", m_running_info.time_total);
            writer.write("time_per_round", m_running_info.time_total / m_running_info.current_round);
            writer.write("num_iteration", (int) m_running_info.current_round);

            if (m_engine_options.IsForceVariant()) {
                writer.write("force_variant", m_engine_options.GetAlgoVariant().ToString());
            }

            if (m_engine_options.IsForceLoadBalancing(MsgPassing::PUSH)) {
                writer.write("force_push_load_balancing",
                 LBToString(m_engine_options.GetLoadBalancing(MsgPassing::PUSH)));
            }

            if (m_engine_options.IsForceLoadBalancing(MsgPassing::PULL)) {
                writer.write("force_pull_load_balancing",
                 LBToString(m_engine_options.GetLoadBalancing(MsgPassing::PULL)));
            }

            if (m_engine_options.GetPriorityType() == Priority::NONE) {
                writer.write("priority_type", "none");
                } else if (m_engine_options.GetPriorityType() == Priority::LOW_HIGH) {
                    writer.write("priority_type", "low_high");
                    writer.write("priority_delta", m_engine_options.GetPriorityThreshold());
                    } else if (m_engine_options.GetPriorityType() == Priority::SAMPLING) {
                        writer.write("priority_type", "sampling");
                        writer.write("cut_threshold", m_engine_options.GetCutThreshold());
                    }

                    writer.write("fused_kernel", m_engine_options.IsFused() ? "YES" : "NO");
                    writer.write("max_iteration_reached",
                     m_running_info.current_round == 1000 ? "YES" : "NO");
                //writer.write("date", get_now());
                writer.write("device", m_dev_props.name);
                writer.write("dataset", FLAGS_graphfile);
                writer.write("nnodes", (int) m_graph_datum->nnodes);
                writer.write("nedges", (int) m_graph_datum->nedges);
                writer.write("algo_type", m_running_info.m_algo_type == AlgoType::TRAVERSAL_SCHEME ? "TRAVERSAL_SCHEME"
                   : "ITERATIVE_SCHEME");
        }

            void PrintInfo() {
                LOG("--------------Overhead--------------\n");
                LOG("Rebuild worklist: %f\n", m_running_info.time_overhead_rebuild_worklist);
                LOG("Priority sample: %f\n", m_running_info.time_overhead_sample);
                LOG("hybrid Worlist: %f\n", m_running_info.time_overhead_hybrid);
                LOG("Unique Worklist: %f\n", m_running_info.time_overhead_wl_unique);
                LOG("--------------Time statistics---------\n");
                LOG("Kernel time: %f\n", m_running_info.time_kernel);
                LOG("Total time: %f\n", m_running_info.time_total);
                LOG("Total rounds: %d\n", m_running_info.current_round);
                LOG("Time/round: %f\n", m_running_info.time_total / m_running_info.current_round);
                LOG("filter_num: %d\n", m_running_info.explicit_num);
                LOG("zerocopy_num: %d\n", m_running_info.zerocopy_num);
                LOG("compaction_num: %d\n", m_running_info.compaction_num);


                LOG("--------------Engine info-------------\n");
                if (m_engine_options.IsForceVariant()) {
                    LOG("Force variant: %s\n", m_engine_options.GetAlgoVariant().ToString().data());
                }

                if (m_engine_options.IsForceLoadBalancing(MsgPassing::PUSH)) {
                    LOG("Force Push Load balancing: %s\n",
                        LBToString(m_engine_options.GetLoadBalancing(MsgPassing::PUSH)).data());
                }

                if (m_engine_options.IsForceLoadBalancing(MsgPassing::PULL)) {
                    LOG("Force Pull Load balancing: %s\n",
                        LBToString(m_engine_options.GetLoadBalancing(MsgPassing::PULL)).data());
                }

                if (m_engine_options.GetPriorityType() == Priority::NONE) {
                    LOG("Priority type: NONE\n");
                    } else if (m_engine_options.GetPriorityType() == Priority::LOW_HIGH) {
                        LOG("Priority type: LOW_HIGH\n");
                        LOG("Priority delta: %f\n", m_engine_options.GetPriorityThreshold());
                        } else if (m_engine_options.GetPriorityType() == Priority::SAMPLING) {
                            LOG("Priority type: Sampling\n");
                            LOG("Cut threshold: %f\n", m_engine_options.GetCutThreshold());
                        }

                        LOG("Fused kernel: %s\n", m_engine_options.IsFused() ? "YES" : "NO");
                        LOG("Max iteration reached: %s\n", m_running_info.current_round == 1000 ? "YES" : "NO");


                        LOG("-------------Misc-------------------\n");
                //LOG("Date: %s\n", get_now().data());
                LOG("Device: %s\n", m_dev_props.name);
                LOG("Dataset: %s\n", FLAGS_graphfile.data());
                LOG("Algo type: %s\n",
                    m_running_info.m_algo_type == AlgoType::TRAVERSAL_SCHEME ? "TRAVERSAL_SCHEME" : "ITERATIVE_SCHEME");
            }

            //traditional graph processing
            void Start(index_t priority_detal = 0) {
                
                GraphDatum &graph_datum = *m_graph_datum;
                graph_datum.priority_detal = priority_detal;
                AlgoVariant next_policy[FLAGS_SEGMENT];
                for(index_t i = 0; i < FLAGS_SEGMENT; i++){
                    next_policy[i] = m_policy_decision_maker.GetInitPolicy();
                }
                bool convergence = false;
                Stopwatch sw_total(true);
                LoadOptions();
                int round = 0;
                while (!convergence) {
                  PreComputationBW();
                  ExecutePolicy_Converge(next_policy);
                //   GatherTransfer();
                  round++;
                  int convergence_check = 0;
                  for(index_t seg_id = 0; seg_id < FLAGS_SEGMENT; seg_id++){
                       if(m_running_info.input_active_count_seg[seg_id] == 0){
                           convergence_check++;
                       }
                   }
                  if(convergence_check == FLAGS_SEGMENT){
                        convergence = true;
                  }
                  if (round == 1000 ) {//FLAGS_max_iteration
                        convergence = true;
                        LOG("Max iterations reached\n");
                  }
                    
                }
               sw_total.stop();
               m_running_info.time_total = sw_total.ms();
               LOG("Iterate all time: %f ms (excluded)\n", sw_total.ms());
            }

            void PrintCacheL1(){
                GraphDatum &graph_datum = *m_graph_datum;
                groute::Stream &stream_s = *m_stream;
                graph_datum.GatherCacheL1();
            }

            void PrintCacheL3(){
                GraphDatum &graph_datum = *m_graph_datum;
                groute::Stream &stream_s = *m_stream;
                graph_datum.GatherCacheL3();
            }
            void PrintCacheNode_verify(){
                GraphDatum &graph_datum = *m_graph_datum;
                groute::Stream &stream_s = *m_stream;
                //把缓存边数据拷贝回CPU，获得host_cache
                graph_datum.GatherCacheL1();
                //把缓存点索引拷贝回CPU获得vertices_【node】.virtual_start
                m_vcsr_dev_graph_allocator->BackNode();
                LOG("--------Verify cache index----------\n");
                const auto &vcsr_graph = m_vcsr_dev_graph_allocator->HostObject();
                uint64_t num_of_dst = 0;
                for(index_t i = 0; i<vcsr_graph.nnodes; i++){
                    if(vcsr_graph.vertices_[i].cache){
                        uint64_t start_h = vcsr_graph.sync_vertices_[i].index;
                        uint32_t hot_deg = vcsr_graph.sync_vertices_[i].degree;
                        uint64_t start_d = vcsr_graph.vertices_[i].virtual_start;
                        uint32_t dev_deg = vcsr_graph.vertices_[i].virtual_degree;
                        uint64_t size = vcsr_graph.vertices_[i].virtual_degree+vcsr_graph.vertices_[i].virtual_start;
                        // printf("vtx %d h_0 %d  h_1 %d h_2 %d h_3 %d\n",i,vcsr_graph.vertices_[i].hotness[0],vcsr_graph.vertices_[i].hotness[1],vcsr_graph.vertices_[i].hotness[2],vcsr_graph.vertices_[i].hotness[3]);
                        // for(auto j = 0; j<  dev_deg ;j++){
                        //     printf("src %d cache_dst %d  ",i,graph_datum.host_cache[start_d+j]);
                        //     printf("hot_dst %d\n",vcsr_graph.edges_[start_h+j]);
                                
                        // }
                        if(size>=graph_datum.num_of_cache){
                            printf("node %d d_start %d size %d\n",i,start_d,dev_deg);
                        }
                        if(start_d>=graph_datum.num_of_cache){
                            printf("node %d d_start  %d\n",i,start_d);
                        }
                        if(hot_deg!=dev_deg)LOG("%d h_deg %d d_deg %d\n",i,vcsr_graph.sync_vertices_[i].degree,vcsr_graph.vertices_[i].virtual_degree);
                        for(auto j = 0; j<  vcsr_graph.sync_vertices_[i].degree ;j++){
                            if(vcsr_graph.edges_[start_h+j]!=graph_datum.host_cache[start_d+j]){
                                printf("src %d cache_dst %d  ",i,graph_datum.host_cache[start_d+j]);
                                printf("hot_dst %d\n",vcsr_graph.edges_[start_h+j]);
                                
                            }
                        }
                    }
                }
                printf("dst为-1的边数量 %d\n",num_of_dst);
                LOG("--------Verify Passed----------\n");
                // m_vcsr_dev_graph_allocator->BackNodeToD();
            }

            void PrintCacheNode_verify_L3(){
                GraphDatum &graph_datum = *m_graph_datum;
                groute::Stream &stream_s = *m_stream;
                //把缓存边数据拷贝回CPU，获得host_cache
                graph_datum.GatherCacheL3();
                //把缓存点索引拷贝回CPU获得vertices_【node】.virtual_start
                m_vcsr_dev_graph_allocator->BackNode();
                LOG("--------Verify cache index----------\n");
                const auto &vcsr_graph = m_vcsr_dev_graph_allocator->HostObject();
                uint64_t num_of_dst = 0;
                for(index_t i = 0; i<vcsr_graph.nnodes; i++){
                    if(vcsr_graph.vertices_[i].cache){
                        uint64_t start_h = vcsr_graph.sync_vertices_[i].index;
                        uint32_t hot_deg = vcsr_graph.sync_vertices_[i].degree;
                        uint64_t start_d = vcsr_graph.vertices_[i].third_start;
                        uint32_t dev_deg = vcsr_graph.vertices_[i].third_degree;
                        uint64_t size = start_d+dev_deg;
                        // printf("vtx %d h_0 %d  h_1 %d h_2 %d h_3 %d\n",i,vcsr_graph.vertices_[i].hotness[0],vcsr_graph.vertices_[i].hotness[1],vcsr_graph.vertices_[i].hotness[2],vcsr_graph.vertices_[i].hotness[3]);
                        // for(auto j = 0; j<  dev_deg ;j++){
                        //     printf("src %d cache_dst %d  ",i,graph_datum.host_cache[start_d+j]);
                        //     printf("hot_dst %d\n",vcsr_graph.edges_[start_h+j]);
                                
                        // }
                        if(size>=graph_datum.num_of_cache){
                            printf("node %d d_start %d size %d\n",i,start_d,dev_deg);
                        }
                        if(start_d>=graph_datum.num_of_cache){
                            printf("node %d d_start  %d\n",i,start_d);
                        }
                        if(hot_deg!=dev_deg)LOG("%d h_deg %d d_deg %d\n",i,hot_deg,dev_deg);
                        for(auto j = 0; j<  vcsr_graph.sync_vertices_[i].degree ;j++){
                            if(vcsr_graph.edges_[start_h+j]!=graph_datum.host_cache_l3[start_d+j]){
                                printf("src %d cache_dst %d  ",i,graph_datum.host_cache_l3[start_d+j]);
                                printf("hot_dst %d\n",vcsr_graph.edges_[start_h+j]);
                                
                            }
                        }
                    }
                }
                printf("dst为-1的边数量 %d\n",num_of_dst);
                LOG("--------Verify Passed----------\n");
                // m_vcsr_dev_graph_allocator->BackNodeToD();
            }

            void Compare_to_cache(){
                GraphDatum &graph_datum = *m_graph_datum;
                graph_datum.GatherCacheL3();
                graph_datum.GatherCacheL1();
                for(uint64_t j =0; j < graph_datum.num_of_cache;j++){
                    if(graph_datum.host_cache[j]!=graph_datum.host_cache_l3[j]){
                        printf("err %d -> %d\n",graph_datum.host_cache[j],graph_datum.host_cache_l3[j]);
                    }
                }
            }

            void PrintCacheNode(){
                GraphDatum &graph_datum = *m_graph_datum;
                groute::Stream &stream_s = *m_stream;
                //把缓存边数据拷贝回CPU，获得host_cache
                graph_datum.GatherCacheL1();
                //把缓存点索引拷贝回CPU获得vertices_【node】.virtual_start
                m_vcsr_dev_graph_allocator->BackNode();
                LOG("--------Verify cache index----------\n");
                const auto &vcsr_graph = m_vcsr_dev_graph_allocator->HostObject();
                uint64_t num_of_dst = 0;
                for(index_t i = 0; i<vcsr_graph.nnodes; i++){
                    for(auto k = 0; k<  vcsr_graph.sync_vertices_[i].degree ;k++){
                        if(vcsr_graph.edges_[vcsr_graph.sync_vertices_[i].index+k]==-1) {
                            printf("wrong src %d dst %d\n",i,vcsr_graph.edges_[vcsr_graph.sync_vertices_[i].index+k]);
                            num_of_dst++;}
                    }
                    if(vcsr_graph.vertices_[i].cache){
                        
                        uint64_t start_h = vcsr_graph.sync_vertices_[i].index;
                        uint32_t hot_deg = vcsr_graph.sync_vertices_[i].degree;
                        uint64_t start_d = vcsr_graph.vertices_[i].virtual_start;
                        uint32_t dev_deg = vcsr_graph.vertices_[i].virtual_degree;
                        uint64_t size = vcsr_graph.vertices_[i].virtual_degree+vcsr_graph.vertices_[i].virtual_start;
                        printf("node %d gpu index  %d gpu deg %d\n",i,start_d,dev_deg);
                        printf("node %d cpu index  %d cpu deg %d\n",i,start_h,hot_deg);
                        for(auto j = 0; j<  dev_deg ;j++){
                            printf("src %d cache_dst %d  ",i,graph_datum.host_cache[start_d+j]);
                            printf("hot_dst %d\n",vcsr_graph.edges_[start_h+j]);
                                
                        }
                        if(size>=graph_datum.num_of_cache){
                            printf("node %d d_start %d size %d\n",i,start_d,dev_deg);
                        }
                        if(start_d>=graph_datum.num_of_cache){
                            printf("node %d d_start  %d\n",i,start_d);
                        }
                        if(hot_deg!=dev_deg)LOG("%d h_deg %d d_deg %d\n",i,vcsr_graph.sync_vertices_[i].degree,vcsr_graph.vertices_[i].virtual_degree);
                        for(auto j = 0; j<  vcsr_graph.sync_vertices_[i].degree ;j++){
                            if(vcsr_graph.edges_[start_h+j]!=graph_datum.host_cache[start_d+j]){
                                printf("src %d cache_dst %d  ",i,graph_datum.host_cache[start_d+j]);
                                printf("hot_dst %d\n",vcsr_graph.edges_[start_h+j]);
                                
                            }
                        }
                    }
                }
                printf("dst为-1的边数量 %d\n",num_of_dst);
                LOG("--------Verify Passed----------\n");
                // m_vcsr_dev_graph_allocator->BackNodeToD();
            }

            void PrintCacheNode_v2(){
                GraphDatum &graph_datum = *m_graph_datum;
                groute::Stream &stream_s = *m_stream;
                // graph_datum.GatherCacheL1();
                // Stopwatch watch(true);
                m_vcsr_dev_graph_allocator->BackNode();
                const auto &vcsr_graph = m_vcsr_dev_graph_allocator->HostObject();
                uint64_t cache_edges = 0;
                uint32_t nodes = 0;
                for(index_t i = 0; i<vcsr_graph.nnodes; i++){
                    if(vcsr_graph.vertices_[i].cache){
                        auto start_h = vcsr_graph.vertices_[i].virtual_start;
                        auto hot_deg = vcsr_graph.vertices_[i].virtual_degree;
                        cache_edges+=hot_deg;
                        nodes++;
                    }
                }
                printf("actually L1 cache nnodes %d  nedges %d\n",nodes,cache_edges);
                // watch.stop();
                // printf("printcache time %f\n",watch.ms());
                // m_vcsr_dev_graph_allocator->BackNodeToD();
            }

            void PrintCacheNode_L3(){
                GraphDatum &graph_datum = *m_graph_datum;
                groute::Stream &stream_s = *m_stream;
                // graph_datum.GatherCacheL1();
                // Stopwatch watch(true);
                m_vcsr_dev_graph_allocator->BackNode();
                const auto &vcsr_graph = m_vcsr_dev_graph_allocator->HostObject();
                uint64_t cache_edges = 0;
                uint32_t nodes = 0;
                for(index_t i = 0; i<vcsr_graph.nnodes; i++){
                    if(vcsr_graph.vertices_[i].cache){
                        auto start_h = vcsr_graph.vertices_[i].third_start;
                        auto hot_deg = vcsr_graph.vertices_[i].third_degree;
                        cache_edges+=hot_deg;
                        nodes++;
                    }
                }
                printf("actually L3 cache nnodes %d  nedges %d\n",nodes,cache_edges);
                // watch.stop();
                // printf("printcache time %f\n",watch.ms());
                // m_vcsr_dev_graph_allocator->BackNodeToD();
            }

            const groute::graphs::host::CSRGraph &CSRGraph() const {
                return m_groute_context->host_graph;
            }

            // const groute::graphs::host::CSRGraph &ValidateGraph() const {
            //     return m_groute_context->host_validate_graph;
            // }
        
            const groute::graphs::host::PMAGraph &PMAGraph() const {
                return m_groute_context->host_pma_small;
            }
            
            const GraphDatum &GetGraphDatum() const {
                return *m_graph_datum;
            }

            void ExecutePolicy_All(AlgoVariant *algo_variant) {
                auto &app_inst = *m_app_inst;
                // auto &vcsr_graph_host = m_vcsr_dev_graph_allocator->HostObject();
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                Stopwatch sw_execution(true);
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
		        uint64_t seg_sedge_csr,seg_nedges_csr;
		        index_t seg_snode,seg_enode;
                // m_groute_context->segment_ct = FLAGS_SEGMENT;
                index_t stream_id;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
    		        seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
    		        seg_enode = m_groute_context->seg_enode[seg_idx];                                    // end node
    		        seg_sedge_csr = m_groute_context->seg_sedge_csr[seg_idx];                            // start edge
    		        seg_nedges_csr = m_groute_context->seg_nedge_csr[seg_idx]; 

    		        stream_id = seg_idx % FLAGS_n_stream;

                    const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                    if(algo_variant[seg_idx] == AlgoVariant::Zero_Copy){
                        m_vcsr_dev_graph_allocator->SwitchZC();
                        zcflag = true;
                        RunSyncPushDDB_ALL(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx,zcflag,
                           vcsr_graph,
                           graph_datum,
                           m_engine_options,
                           stream[stream_id]); 
                    }
               }
               for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }
               sw_execution.stop();
               LOG("part add edge time: %f ms (excluded)\n", sw_execution.ms());
            //    sw_round.stop();

            }

            //memory free policy for gpu incremental graph processing
            void ExecutePolicy_MF(AlgoVariant *algo_variant,int *type_device) {
                LOG("------MF PUSH------\n");
                auto &app_inst = *m_app_inst;
                // auto &vcsr_graph_host = m_vcsr_dev_graph_allocator->HostObject();
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
		        uint64_t seg_sedge_csr,seg_nedges_csr;
		        index_t seg_snode,seg_enode;
                index_t seg_idx_new;

                m_groute_context->segment_ct = FLAGS_SEGMENT;
                index_t stream_id;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){

    		        seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
    		        seg_enode = m_groute_context->seg_enode[seg_idx];                                    // end node
    		        seg_sedge_csr = m_groute_context->seg_sedge_csr[seg_idx];                            // start edge
    		        seg_nedges_csr = m_groute_context->seg_nedge_csr[seg_idx]; 

    		        stream_id = seg_idx % FLAGS_n_stream;

                    const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                    if(algo_variant[seg_idx] == AlgoVariant::Zero_Copy){
                        m_vcsr_dev_graph_allocator->SwitchZC();
                        zcflag = true;
                        RunSyncPushDDBAmend(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx,zcflag,
                           vcsr_graph,type_device,
                           graph_datum,
                           m_engine_options,
                           stream[stream_id]); 
                    }
               }
               for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }
            }

            //Cache filling
            void ExecutePolicy_MF_Cache(AlgoVariant *algo_variant,int *type_device) {
                auto &app_inst = *m_app_inst;
                // auto &vcsr_graph_host = m_vcsr_dev_graph_allocator->HostObject();
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
		        uint64_t seg_sedge_csr,seg_nedges_csr;
		        index_t seg_snode,seg_enode;
                index_t seg_idx_new;

                m_groute_context->segment_ct = FLAGS_SEGMENT;
                index_t stream_id;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){

    		        seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
    		        seg_enode = m_groute_context->seg_enode[seg_idx];                                    // end node
    		        seg_sedge_csr = m_groute_context->seg_sedge_csr[seg_idx];                            // start edge
    		        seg_nedges_csr = m_groute_context->seg_nedge_csr[seg_idx]; 

    		        stream_id = seg_idx % FLAGS_n_stream;

                    const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                    if(algo_variant[seg_idx] == AlgoVariant::Zero_Copy){
                        m_vcsr_dev_graph_allocator->SwitchZC();
                        zcflag = true;
                        RunSyncPushDDBCache(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx,zcflag,
                           vcsr_graph,type_device,
                           graph_datum,
                           m_engine_options,
                           stream[stream_id]); 
                    }
               }
               for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }
            }

            
            void ExecutePolicy_Com(AlgoVariant *algo_variant) {
                auto &app_inst = *m_app_inst;
                // auto &vcsr_graph_host = m_vcsr_dev_graph_allocator->HostObject();
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
		        uint64_t seg_sedge_csr,seg_nedges_csr;
		        index_t seg_snode,seg_enode;
                Stopwatch sw_execution(true);
                index_t stream_id;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
    		        seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
    		        seg_enode = m_groute_context->seg_enode[seg_idx];                                    // end node
    		        seg_sedge_csr = m_groute_context->seg_sedge_csr[seg_idx];                            // start edge
    		        seg_nedges_csr = m_groute_context->seg_nedge_csr[seg_idx];
    		        stream_id = seg_idx % FLAGS_n_stream;
                    const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                    if(algo_variant[seg_idx] == AlgoVariant::Zero_Copy){
                        m_vcsr_dev_graph_allocator->SwitchZC();
                        zcflag = true;
                        RunSyncCom(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx,zcflag,
                           vcsr_graph,
                           graph_datum,
                           m_engine_options,
                           stream[stream_id]); 
                    }
               }
               for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }
                sw_execution.stop();
               LOG("flush cache(compact time): %f ms (excluded)\n", sw_execution.ms());
            }


            void get_update_file(){
                m_load_update = std::unique_ptr<Loader>(new Loader(FLAGS_update_size,FLAGS_weight));
                Loader &load_update = *m_load_update;
                load_update.ReadWeightList(FLAGS_updatefile);
                this->added_edges_h = (WEdge*)malloc(sizeof(WEdge)*(load_update.m_add_size));
                this->del_edges_h = (WEdge*)malloc(sizeof(WEdge)*(load_update.m_del_size));
                GROUTE_CUDA_CHECK(cudaHostRegister((void *)(this->added_edges_h), sizeof(WEdge) * (load_update.m_add_size), cudaHostRegisterMapped));

                GROUTE_CUDA_CHECK(cudaHostRegister((void *)(this->del_edges_h), sizeof(WEdge) * (load_update.m_del_size), cudaHostRegisterMapped));

                GROUTE_CUDA_CHECK(cudaMalloc(&(this->work_size_d), 2 * sizeof(uint32_t)));
                GROUTE_CUDA_CHECK(cudaHostRegister((void *)this->type, sizeof(int) * 2, cudaHostRegisterMapped));
            }

            void add_edge(std::pair<index_t,index_t>& local_begin,index_t& NumOfSnapShots){
                Loader &load_update = *m_load_update;

                update_tree_add(local_begin,NumOfSnapShots);
            }

            void compute_hot_vertices(){
                GraphDatum &graph_datum = *m_graph_datum;
                auto &app_inst = *m_app_inst;
                groute::Stream &stream_s = *m_stream;
                const auto &work_source = groute::dev::WorkSourceRange<index_t>(0, graph_datum.nnodes); 
                const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                const auto &hvcsr = m_vcsr_dev_graph_allocator->HostObject();
                dim3 grid_dims, block_dims;
                KernelSizing(grid_dims, block_dims, work_source.get_size());
                kernel::comp_hotness<< < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (app_inst,
                    vcsr_graph,
                    work_source,
                    graph_datum.d_hotness);
                graph_datum.sort_vtx_by_hotness();

                cudaDeviceSynchronize();
                graph_datum.CompareDeviceResult();
                cudaDeviceSynchronize();
            }

            void add_edge_pr(std::pair<index_t,index_t>& local_begin,index_t &NumOfSnapShots){
                Loader &load_update = *m_load_update;
                groute::graphs::host::PMAGraph &vcsr_graph  = m_vcsr_dev_graph_allocator->m_origin_graph;
                for(index_t i = local_begin.first; i < local_begin.first+load_update.m_batch_size[NumOfSnapShots].first; i++){
                    index_t src_add = load_update.added_edges_w[i].u;
                    index_t dst_add = load_update.added_edges_w[i].v;
                    this->added_edges_h[i].u = src_add;
                    this->added_edges_h[i].v = dst_add;
                    this->added_edges_h[i].w = (dst_add+src_add)%128 + 1;
                    vcsr_graph.insert(src_add, dst_add, (src_add + dst_add)%128 + 1);
                }
            }

            void del_edge_pr(std::pair<index_t,index_t>& local_begin,index_t &NumOfSnapShots){
                Loader &load_update = *m_load_update;
                groute::graphs::host::PMAGraph &vcsr_graph  = m_vcsr_dev_graph_allocator->m_origin_graph;
                index_t size = load_update.m_batch_size[NumOfSnapShots].second;
                for(index_t i = local_begin.second; i < local_begin.second+ size; i++){
                    index_t src_del = load_update.deleted_edges_w[i].u;
                    index_t dst_del = load_update.deleted_edges_w[i].v;
                    vcsr_graph.del_edge(src_del, dst_del, (src_del + dst_del)%128+1);
                }
            }

            void del_edge(std::pair<index_t,index_t>& local_begin,index_t& NumOfSnapShots){
                LOG("----------Batch: %d---------\n",NumOfSnapShots);
                Loader &load_update = *m_load_update;
                groute::graphs::host::PMAGraph &vcsr_graph  = m_vcsr_dev_graph_allocator->m_origin_graph;
                index_t size = load_update.m_batch_size[NumOfSnapShots].second;
                for(index_t i = local_begin.second; i < local_begin.second+size; i++){
                    index_t src_del = load_update.deleted_edges_w[i].u;
                    index_t dst_del = load_update.deleted_edges_w[i].v;
                    this->del_edges_h[i].u = src_del;
                    this->del_edges_h[i].v = dst_del;
                    this->del_edges_h[i].w = (src_del+dst_del)%128+1;
                }
                update_tree_del(local_begin,NumOfSnapShots);
            }

            void read_del(std::pair<index_t,index_t> &local_begin,index_t &NumOfSnapShots){
                auto &app_inst = *m_app_inst;
                Loader &load_update = *m_load_update;
                GraphDatum &graph_datum = *m_graph_datum;
                auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                groute::Stream &stream_s = *m_stream;
                index_t size = load_update.m_batch_size[NumOfSnapShots].second;
                for(index_t i = local_begin.second; i < local_begin.second+size; i++){
                    index_t src_del = load_update.deleted_edges_w[i].u;
                    index_t dst_del = load_update.deleted_edges_w[i].v;
                    this->del_edges_h[i].u = src_del;
                    this->del_edges_h[i].v = dst_del;
                    this->del_edges_h[i].w = (src_del+dst_del)%128+1;
                }
                // LOG("DEBUG pr 1.3 \n");

                dim3 grid_dims, block_dims;
                uint32_t work_size[2];
                index_t start = local_begin.second;
                work_size[0] = start;
                work_size[1] = size;
                // LOG("DEBUG pr 1.3.1 \n");
                LOG("del start %d \n",work_size[0]);
                LOG("del size %d \n",work_size[1]);
                GROUTE_CUDA_CHECK(cudaHostGetDevicePointer((void **)&(this->del_edges_d), (void *)(this->del_edges_h), 0));
                GROUTE_CUDA_CHECK(cudaMemcpy(this->work_size_d, &work_size[0], 2 * sizeof(uint32_t),cudaMemcpyHostToDevice));
                // LOG("DEBUG pr 1.3.2 \n");

                KernelSizing(grid_dims, block_dims, size);
                // LOG("DEBUG pr 1.3.3 \n");
                // bool del = true;
                kernel::reset_pr_del_edges<<< grid_dims, block_dims, 0, stream_s.cuda_stream >>>(app_inst,vcsr_graph,
                this->del_edges_d,
                this->work_size_d);
                stream_s.Sync();
                // LOG("DEBUG pr 1.3.5 \n");
                
            }

            void read_add(std::pair<index_t,index_t> &local_begin,index_t &NumOfSnapShots){
                auto &app_inst = *m_app_inst;
                Loader &load_update = *m_load_update;
                groute::Stream &stream_s = *m_stream;
                GraphDatum &graph_datum = *m_graph_datum;
                auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                index_t size = load_update.m_batch_size[NumOfSnapShots].first;
                for(index_t i = local_begin.first; i < local_begin.first+size; i++){
                    index_t src_add = load_update.added_edges_w[i].u;
                    index_t dst_add = load_update.added_edges_w[i].v;
                    this->added_edges_h[i].u = src_add;
                    this->added_edges_h[i].v = dst_add;
                    this->added_edges_h[i].w = (src_add+dst_add)%128+1;
                    // printf("host add edge %d %d\n",src_add,dst_add);
                }
                // LOG("DEBUG pr 1.2 \n");
                GROUTE_CUDA_CHECK(cudaHostGetDevicePointer((void **)&(this->added_edges_d), (void *)(this->added_edges_h), 0));
                dim3 grid_dims, block_dims;
                uint32_t work_size[2];
                work_size[0] = local_begin.first;
                work_size[1] = size;
                // LOG("add start %d \n",work_size[0]);
                // LOG("add size %d \n",work_size[1]);
                bool del =false;
                GROUTE_CUDA_CHECK(cudaMemcpy(this->work_size_d, &work_size[0], 2 * sizeof(uint32_t),cudaMemcpyHostToDevice));
                KernelSizing(grid_dims, block_dims, size);
                kernel::reset_pr_del_edges<<< grid_dims, block_dims, 0, stream_s.cuda_stream >>>(app_inst,vcsr_graph,this->added_edges_d,this->work_size_d);
                stream_s.Sync();
                // LOG("DEBUG pr 1.2.5 \n");
            }

            void Cancelation(std::pair<index_t,index_t>& local_begin,index_t& NumOfSnapShots){
                //回收
                LOG("====this is %d batch====\n",NumOfSnapShots);
                auto &app_inst = *m_app_inst;
                GraphDatum &graph_datum = *m_graph_datum;
                Loader &load_update = *m_load_update;
                groute::Stream &stream_s = *m_stream;
                auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                type[0] = -1;
                
                GROUTE_CUDA_CHECK(cudaHostGetDevicePointer((void **)&this->type_device, (void *)this->type, 0));
                index_t seg_snode,seg_enode;
                index_t stream_id;
                // LOG("cancel: -----------1-------\n");
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
                    seg_snode = m_groute_context->seg_snode[seg_idx];
                    seg_enode = m_groute_context->seg_enode[seg_idx];  
                        RebuildWorklist_AllVertices(app_inst,
                        graph_datum,
                        stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                }
                // std::cout<<"pr active all node "<<vcsr_graph.nnodes<<std::endl;
                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                    stream[stream_idx].Sync();
                }
                AlgoVariant next_policy[FLAGS_SEGMENT];
                for(index_t i = 0; i < FLAGS_SEGMENT; i++){
                    next_policy[i] = m_policy_decision_maker.GetInitPolicy();
                }
                // LOG("cancel: -----------2-----\n");
                // PrintCacheNode();
                Stopwatch sw_execution(true);
                ExecutePolicy_MF(next_policy,type_device);
                // GatherTransfer();
                count();
                sw_execution.stop();
                LOG("取消时间: %f ms (excluded)\n", sw_execution.ms());
                // PrintCacheNode();

                // printf("---------------for cache------------------\n");
                // for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                //     stream_id = seg_idx % FLAGS_n_stream;
                //     seg_snode = m_groute_context->seg_snode[seg_idx];
                //     seg_enode = m_groute_context->seg_enode[seg_idx];  
                //         RebuildWorklist_AllVertices(app_inst,
                //         graph_datum,
                //         stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                // }

                // for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                //     stream[stream_idx].Sync();
                // }
                // ExecutePolicy_MF_Cache(next_policy,type_device);
                 
               
            }
            void ResetCacheMiss(){
                GraphDatum &graph_datum = *m_graph_datum;
                groute::Stream &stream_s = *m_stream;
                const auto &work_source = groute::dev::WorkSourceRange<index_t>(0, graph_datum.nnodes); 
                dim3 grid_dims, block_dims;
                KernelSizing(grid_dims, block_dims, work_source.get_size());
            kernel::reset_cache
            << < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (
                work_source,
                graph_datum.count_gpu);
            stream_s.Sync();
                graph_datum.ResetCacheMiss();
            }
            
            void Resettransfer(){
                GraphDatum &graph_datum = *m_graph_datum;
                groute::Stream &stream_s = *m_stream;
                const auto &work_source = groute::dev::WorkSourceRange<index_t>(0, graph_datum.nnodes); 
                dim3 grid_dims, block_dims;
                KernelSizing(grid_dims, block_dims, work_source.get_size());
            kernel::reset_cache
            << < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (
                work_source,
                graph_datum.total_act_d);
            stream_s.Sync();
                graph_datum.Resettransfer();
            }

            void GatherCacheMiss(){
                GraphDatum &graph_datum = *m_graph_datum;
                groute::Stream &stream_s = *m_stream;
                auto &vcsr_graph = m_vcsr_dev_graph_allocator->HostObject();
                graph_datum.GatherCacheMiss();
                uint64_t cache_miss =0;
                // UINT64_MAX
                for(index_t i=0;i<vcsr_graph.nnodes;i++){
                    cache_miss += graph_datum.count_cpu[i];
                    graph_datum.count_cpu[i] = 0;
                }
                printf("缓存未命中数量 %llu\n",cache_miss);
                ResetCacheMiss();
            }

            void GatherTransfer(){
                GraphDatum &graph_datum = *m_graph_datum;
                groute::Stream &stream_s = *m_stream;
                auto &vcsr_graph = m_vcsr_dev_graph_allocator->HostObject();
                graph_datum.Gathertransfer();
                uint64_t transfer =0;
                // UINT64_MAX
                for(index_t i=0;i<vcsr_graph.nnodes;i++){
                    transfer += graph_datum.total_act[i];
                    graph_datum.total_act[i] = 0;
                }
                printf("传输总量 %llu\n",transfer);
                Resettransfer();;
            }
            void Compensate(std::pair<index_t,index_t>& local_begin,index_t& NumOfSnapShots){
                auto &app_inst = *m_app_inst;
                GraphDatum &graph_datum = *m_graph_datum;
                Loader &load_update = *m_load_update;
                groute::Stream &stream_s = *m_stream;
                index_t seg_snode,seg_enode;
                index_t stream_id;
                AlgoVariant next_policy[FLAGS_SEGMENT];
                for(index_t i = 0; i < FLAGS_SEGMENT; i++){
                    next_policy[i] = m_policy_decision_maker.GetInitPolicy();
                }
                
                auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();

                //update graph on the cpu
                read_del(local_begin,NumOfSnapShots);
                cudaDeviceSynchronize();
                read_add(local_begin,NumOfSnapShots);
                cudaDeviceSynchronize();
                
                add_edge_pr(local_begin,NumOfSnapShots);
                del_edge_pr(local_begin,NumOfSnapShots);
                local_begin.second += load_update.m_batch_size[NumOfSnapShots].second;
                local_begin.first += load_update.m_batch_size[NumOfSnapShots].first;

                m_vcsr_dev_graph_allocator->ReloadAllocator();
                type[0] = 1;
                float time_total = 0;
                GROUTE_CUDA_CHECK(cudaHostGetDevicePointer((void **)&this->type_device, (void *)this->type, 0));
                
                //incremental computation
                Stopwatch sw_execution(true);
                ExecutePolicy_MF(next_policy,type_device);
                // GatherTransfer();
                // PostComputationBW();
                sw_execution.stop();
                time_total +=sw_execution.ms();
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
                    seg_snode = m_groute_context->seg_snode[seg_idx];
                    seg_enode = m_groute_context->seg_enode[seg_idx];  
                        RebuildArrayWorklistINC(app_inst,
                        graph_datum,
                        stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                }
                for(index_t stream_idx = 0; stream_idx <  FLAGS_n_stream ; stream_idx++){
                    stream[stream_idx].Sync();
                }
                LOG("------------PR mf compensate-----------\n");
                count();
                for(index_t stream_idx = 0; stream_idx <  FLAGS_n_stream ; stream_idx++){
                    stream[stream_idx].Sync();
                }
                bool convergence = false;
                int current_rount =0 ;
                // m_running_info.current_round = 0;
                while(!convergence){
                  Stopwatch sw_execution_round(true);
                  ExecutePolicy_MF_Converge(next_policy);
                //   GatherTransfer();
                  current_rount++;
                  sw_execution_round.stop();
                  time_total +=sw_execution_round.ms();	   
                  PostComputationBW_Inc();
                  if (current_rount == 100 ) {//FLAGS_max_iteration
                        convergence = true;
                        LOG("Max iterations reached\n");
                  } 
                }
                // sw_execution.stop();

                LOG("迭代时间: %f ms (excluded)\n", time_total);
                LOG("round num : %d  (excluded)\n", current_rount);
                // PrintCacheNode_v2();
                // GatherCacheMiss();
                // GatherTransfer();
                // PrintCacheNode_v2();
            }


            void compact_cache() {
                groute::Stream &stream_s = *m_stream;
                auto &app_inst = *m_app_inst;
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
		        uint64_t seg_sedge_csr,seg_nedges_csr;
		        index_t seg_snode,seg_enode;
                Stopwatch sw_execution(true);
                index_t stream_id;
                const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
                    seg_snode = m_groute_context->seg_snode[seg_idx];
                    seg_enode = m_groute_context->seg_enode[seg_idx];  
                        RebuildArrayWorklist_identify(app_inst,vcsr_graph,
                        graph_datum,
                        stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                }
                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }
               cudaDeviceSynchronize();
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
    		        seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
    		        seg_enode = m_groute_context->seg_enode[seg_idx];                                    // end node
    		        seg_sedge_csr = m_groute_context->seg_sedge_csr[seg_idx];                            // start edge
    		        seg_nedges_csr = m_groute_context->seg_nedge_csr[seg_idx];
    		        stream_id = seg_idx % FLAGS_n_stream;
                    
                    m_vcsr_dev_graph_allocator->SwitchZC();
                    zcflag = true;
                    RunSyncCom(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx,zcflag,
                        vcsr_graph,
                        graph_datum,
                        m_engine_options,
                        stream[stream_id]);
               }
               for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }
               cudaDeviceSynchronize();
               m_vcsr_dev_graph_allocator->HostRiverElement();
               dim3 grid_dims, block_dims;
               const auto &work_source = groute::dev::WorkSourceRange<index_t>(0, graph_datum.num_of_cache); 
               KernelSizing(grid_dims, block_dims, work_source.get_size());
                kernel::copy_cache<< < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (graph_datum.cache_edges_l1,graph_datum.cache_edges_com,work_source);
                stream_s.Sync();
                cudaDeviceSynchronize();
                const auto &work_source_flush_vertex = groute::dev::WorkSourceRange<index_t>(0, graph_datum.nnodes); 
                KernelSizing(grid_dims, block_dims, work_source_flush_vertex.get_size());
                kernel::copy_index<< < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (app_inst,
                    vcsr_graph,
                    work_source_flush_vertex);
                    stream_s.Sync();
                sw_execution.stop();
               LOG("合并缓存时间 cache(compact time): %f ms (excluded)\n", sw_execution.ms());
            }

            void evication_cache(){
                groute::Stream &stream_s = *m_stream;
                auto &app_inst = *m_app_inst;
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
                
		        uint64_t seg_sedge_csr,seg_nedges_csr;
		        index_t seg_snode,seg_enode;
                Stopwatch sw_execution(true);
                index_t stream_id;
                const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                const auto &work_source = groute::dev::WorkSourceRange<index_t>(0, graph_datum.nnodes); 
                dim3 grid_dims, block_dims;
                KernelSizing(grid_dims, block_dims, work_source.get_size());
                kernel::flush_cache<< < grid_dims, block_dims, 0, stream_s.cuda_stream >> > (app_inst,
                    vcsr_graph,
                    work_source,
                    graph_datum.d_hotness);
                stream_s.Sync();
                sw_execution.stop();
                LOG("缓存逐出时间 %f\n",sw_execution.ms());
            }


            void ExecutePolicy_MF_Converge(AlgoVariant *algo_variant) {
                // printf("conver iter\n");
                auto &app_inst = *m_app_inst;
                auto &vcsr_graph_host = m_vcsr_dev_graph_allocator->HostObject();
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                Stopwatch sw_execution(true);
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
		        uint64_t seg_sedge_csr,seg_nedges_csr;
		        index_t seg_snode,seg_enode;

                m_groute_context->segment_ct = FLAGS_SEGMENT;
                index_t seg_exc = 0;
                index_t stream_id;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
    		        seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
    		        seg_enode = m_groute_context->seg_enode[seg_idx];                                    // end node
    		        seg_sedge_csr = m_groute_context->seg_sedge_csr[seg_idx];                            // start edge
    		        seg_nedges_csr = m_groute_context->seg_nedge_csr[seg_idx]; 
    		        stream_id = seg_idx % FLAGS_n_stream;
                    const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                    if(algo_variant[seg_idx] == AlgoVariant::Zero_Copy){
                        m_vcsr_dev_graph_allocator->SwitchZC();
                        zcflag = true;
                        RunSyncPushDDB_MFC(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx,zcflag,
                           vcsr_graph,
                           graph_datum,
                           m_engine_options,
                           stream[stream_id]); 
                    }
                    // act_num+= graph_datum.m_wl_array_in[seg_idx];
               }
               for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }

            //    PostComputationBW_Inc();
               
            //    sw_round.stop();

            }
            void PostComputationBW_Inc() {
                // printf("------------PostComputationBW-----------\n");
                int dev_id = 0;
                const groute::Stream &stream_seg = m_groute_context->CreateStream(dev_id);
                GraphDatum &graph_datum = *m_graph_datum;
                AppImplDeviceObject &app_inst = *m_app_inst;
                m_running_info.current_round = m_graph_datum->m_current_round.get_val_D2H();

                Stopwatch sw_unique(true);

                index_t seg_snode,seg_enode;
                index_t stream_id;

                Stopwatch sw_rebuild(true);
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
			        seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
			        seg_enode = m_groute_context->seg_enode[seg_idx];  
			        RebuildArrayWorklistINC(app_inst,
                        graph_datum,
                        stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                }

                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                    stream[stream_idx].Sync();
                }
                sw_rebuild.stop();
                uint64_t active_count_overall = 0;
                m_running_info.time_overhead_rebuild_worklist += sw_rebuild.ms();
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream; 
                    index_t active_count = graph_datum.m_wl_array_in_seg[seg_idx].GetCount(stream[stream_id]);
		            graph_datum.seg_active_num[seg_idx] = active_count;

		            m_running_info.input_active_count_seg[seg_idx] = active_count;

		            uint32_t work_size = active_count;
		            dim3 grid_dims, block_dims;
                    // active_count_overall+=work_size;

                }
                // printf("Round act node %d\n",active_count_overall);
                sw_unique.stop();
                m_running_info.time_overhead_wl_unique += sw_unique.ms();

          }
            
            void del_topo(index_t &local_begin){
                Loader &load_update = *m_load_update;
                groute::graphs::host::PMAGraph &vcsr_graph  = m_vcsr_dev_graph_allocator->m_origin_graph;
                // index_t size = load_update.m_batch_size/(2*NumOfSnapShots);
                index_t size = load_update.m_del_size;
                for(index_t i = 0; i < size; i++){
                    index_t src_del = load_update.deleted_edges_w[i].u;
                    index_t dst_del = load_update.deleted_edges_w[i].v;
                    vcsr_graph.del_edge(src_del, dst_del, (src_del + dst_del)%128+1);
                }
            }

            void reset_delta_vertices(){
                auto &app_inst = *m_app_inst;
                GraphDatum &graph_datum = *m_graph_datum;
                index_t seg_snode,seg_enode;
                index_t stream_id;
                auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
                    seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
                    seg_enode = m_groute_context->seg_enode[seg_idx];  
                        RebuildArrayWorklist_reset_delta(app_inst,vcsr_graph,
                        graph_datum,
                        stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                }    
                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                   stream[stream_idx].Sync();
                }
            }

            void update_tree_add(std::pair<index_t,index_t> &local_begin,index_t &NumOfSnapShots){
                // LOG("-----perform add-----\n");
                auto &app_inst = *m_app_inst;
                GraphDatum &graph_datum = *m_graph_datum;
                Loader &load_update = *m_load_update;
                groute::Stream &stream_s = *m_stream;
                auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                //init the zero copy policy
                AlgoVariant next_policy[FLAGS_SEGMENT];
                for(index_t i = 0; i < FLAGS_SEGMENT; i++){
                    next_policy[i] = m_policy_decision_maker.GetInitPolicy();
                }
                // just update the vertex
                read_del(local_begin,NumOfSnapShots);
                cudaDeviceSynchronize();
                read_add(local_begin,NumOfSnapShots);
                cudaDeviceSynchronize();
                add_edge_pr(local_begin,NumOfSnapShots);
                del_edge_pr(local_begin,NumOfSnapShots);
                local_begin.second += load_update.m_batch_size[NumOfSnapShots].second;
                local_begin.first += load_update.m_batch_size[NumOfSnapShots].first;
                m_vcsr_dev_graph_allocator->ReloadAllocator();
                index_t seg_snode,seg_enode;
                index_t stream_id;
                float add_time = 0;
                Stopwatch sw_load(true);
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
                    seg_snode = m_groute_context->seg_snode[seg_idx];
                    seg_enode = m_groute_context->seg_enode[seg_idx];  
                        RebuildWorklist_AllVertices(app_inst,
                        graph_datum,
                        stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                }
                std::cout<<"sssp active all node "<<vcsr_graph.nnodes<<std::endl;
                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                    stream[stream_idx].Sync();
                }
                // LOG("DEBUG1 \n");
                // PreComputationBW();
                ExecutePolicy_All(next_policy);
                // GatherTransfer();
                sw_load.stop();
                add_time+=sw_load.ms();
                // GatherCacheMiss();
                // GatherTransfer();
                cudaDeviceSynchronize();
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
			        seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
			        seg_enode = m_groute_context->seg_enode[seg_idx];  
			        RebuildArrayWorklist(app_inst,
                        graph_datum,
                        stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                }
                // LOG("DEBUG2 \n");
                bool convergence = false;
                m_running_info.current_round = 0;
                Stopwatch sw_con(true);
                while(!convergence){
                //   PreComputationBW();
                  ExecutePolicy_Converge(next_policy);
                //   GatherTransfer();
                //   ExecutePolicy_SC(next_policy);
                  int convergence_check = 0;
                  for(index_t seg_id = 0; seg_id < FLAGS_SEGMENT; seg_id++){
                       if(m_running_info.input_active_count_seg[seg_id] == 0){
                           convergence_check++;
                       }
                   }
                  if(convergence_check == FLAGS_SEGMENT){
                        convergence = true;
                  }
                  if (m_running_info.current_round == 100 ) {//FLAGS_max_iteration
                        convergence = true;
                        LOG("Max iterations reached\n");
                  } 
                }
                sw_con.stop();
                add_time+=sw_con.ms();
                LOG("total add time: %f ms (excluded)\n", add_time);
                // GatherCacheMiss();
                // GatherTransfer();
            }

            void update_tree_del(std::pair<index_t,index_t> &local_begin,index_t &NumOfSnapShots){
                auto &app_inst = *m_app_inst;
                GraphDatum &graph_datum = *m_graph_datum;
                Loader &load_update = *m_load_update;
                groute::Stream &stream_s = *m_stream;
                const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                AlgoVariant next_policy[FLAGS_SEGMENT];
                for(index_t i = 0; i < FLAGS_SEGMENT; i++){
                    next_policy[i] = m_policy_decision_maker.GetInitPolicy();
                }
                // m_vcsr_dev_graph_allocator->ReloadAllocator();
                GROUTE_CUDA_CHECK(cudaHostGetDevicePointer((void **)&(this->del_edges_d), (void *)(this->del_edges_h), 0));
                dim3 grid_dims, block_dims;
                uint32_t work_size[2];
                index_t start = local_begin.second;
                index_t size = load_update.m_batch_size[NumOfSnapShots].second;
                work_size[0] = start;
                work_size[1] = size;
                GROUTE_CUDA_CHECK(cudaMemcpy(this->work_size_d, &work_size[0], 2 * sizeof(uint32_t),cudaMemcpyHostToDevice));
                //(1) you need reset the parent, value, buffer of deleted edge dst, 如果删除边的源点是终点的parent
                KernelSizing(grid_dims, block_dims, size);
                // Stopwatch sw_load(true);
                Stopwatch sw_del(true);
                kernel::reset_del_edges<<< grid_dims, block_dims, 0, stream_s.cuda_stream >>>(app_inst,vcsr_graph,graph_datum.GetParentDeviceObject(),graph_datum.GetValueDeviceObject(),graph_datum.GetBufferDeviceObject(),graph_datum.cache_edges_l1,graph_datum.cache_edges_l2,this->del_edges_d,this->work_size_d,graph_datum.m_node_reset_datum);
                stream_s.Sync(); 
                cudaDeviceSynchronize();
                index_t seg_snode,seg_enode;
                index_t stream_id;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
                    seg_snode = m_groute_context->seg_snode[seg_idx];
                    seg_enode = m_groute_context->seg_enode[seg_idx];  
                        // RebuildArrayWorklistAdd(app_inst,vcsr_graph,
                        RebuildArrayWorklistDel(app_inst,vcsr_graph,
                        graph_datum,
                        stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                }
                bool convergence = false;
                m_running_info.current_round = 0;
                while(!convergence){
                  PreComputationBW();
                  ExecutePolicy_del_con(next_policy);
                //   GatherTransfer();
                  int convergence_check = 0;
                  for(index_t seg_id = 0; seg_id < FLAGS_SEGMENT; seg_id++){
                       if(m_running_info.input_active_count_seg[seg_id] == 0){
                           convergence_check++;
                       }
                   }
                  if(convergence_check == FLAGS_SEGMENT){
                        convergence = true;
                  }
                  if (m_running_info.current_round == 100 ) {//FLAGS_max_iteration
                        convergence = true;
                        LOG("Max iterations reached\n");
                  } 
                }
                sw_del.stop();
                LOG("删除 time: %f ms (excluded)\n", sw_del.ms());
                // GatherCacheMiss();
                // GatherTransfer();
                // local_begin.second += load_update.m_batch_size[NumOfSnapShots].second;
                // NumOfSnapShots++;
            }

            void GatherValue() {
                return m_graph_datum->GatherValue();
            }

            void GatherReset() {
                // return m_graph_datum->GatherValue();
                return m_graph_datum->GatherReset();
                // m_graph_datum->GatherBuffer();
                // auto &ranks=this->GetGraphDatum().host_value;
                // auto &deltas=this->GetGraphDatum().host_buffer;
                // for(index_t i = 0 ; i < 100; i++){
                //     printf("%d rank %f delta %f\n",i,ranks[i],deltas[i]);
                // }

            }

            void GatherParent() {
                return m_graph_datum->GatherParent();
            }

            void GatherBuffer() {
                return m_graph_datum->GatherBuffer();
            }

            void GatherLevel(){
                return m_graph_datum->GatherLevel();
            }

            groute::graphs::dev::CSRGraph CSRDeviceObject() const {
                return m_csr_dev_graph_allocator->DeviceObject();
            }

            const groute::Stream &getStream() const {
                return *m_stream;
            }

            private:
            void LoadOptions() {
                if (!m_engine_options.IsForceLoadBalancing(MsgPassing::PUSH)) {
                    if (FLAGS_lb_push.size() == 0) {
                        if (m_groute_context->host_pma_small.avg_degree() >= 0) { //all FINE_GRAINED
                            m_engine_options.SetLoadBalancing(MsgPassing::PUSH, LoadBalancing::FINE_GRAINED);
                        }
                        } else {
                            if (FLAGS_lb_push == "none") {
                                m_engine_options.SetLoadBalancing(MsgPassing::PUSH, LoadBalancing::NONE);
                                } else if (FLAGS_lb_push == "coarse") {
                                    m_engine_options.SetLoadBalancing(MsgPassing::PUSH, LoadBalancing::COARSE_GRAINED);
                                    } else if (FLAGS_lb_push == "fine") {
                                        m_engine_options.SetLoadBalancing(MsgPassing::PUSH, LoadBalancing::FINE_GRAINED);
                                        } else if (FLAGS_lb_push == "hybrid") {
                                            m_engine_options.SetLoadBalancing(MsgPassing::PUSH, LoadBalancing::HYBRID);
                                            } else {
                                                fprintf(stderr, "unknown push load-balancing policy");
                                                exit(1);
                                            }
                                        }
                                    }

                if (!m_engine_options.IsForceLoadBalancing(MsgPassing::PULL)) {
                    if (FLAGS_lb_pull.size() == 0) {
                        if (m_groute_context->host_pma_small.avg_degree() >= 5) {
                            m_engine_options.SetLoadBalancing(MsgPassing::PULL, LoadBalancing::FINE_GRAINED);
                        }
                        } else {
                            if (FLAGS_lb_pull == "none") {
                                m_engine_options.SetLoadBalancing(MsgPassing::PULL, LoadBalancing::NONE);
                                } else if (FLAGS_lb_pull == "coarse") {
                                    m_engine_options.SetLoadBalancing(MsgPassing::PULL, LoadBalancing::COARSE_GRAINED);
                                    } else if (FLAGS_lb_pull == "fine") {
                                        m_engine_options.SetLoadBalancing(MsgPassing::PULL, LoadBalancing::FINE_GRAINED);
                                        } else if (FLAGS_lb_pull == "hybrid") {
                                            m_engine_options.SetLoadBalancing(MsgPassing::PULL, LoadBalancing::HYBRID);
                                            } else {
                                                fprintf(stderr, "unknown pull load-balancing policy");
                                                exit(1);
                                            }
                                        }
                                    }

                                    if (FLAGS_alpha == 0) {
                                        fprintf(stderr, "Warning: alpha = 0, A general method AsyncPushDD is used\n");
                                        m_engine_options.ForceVariant(AlgoVariant::ASYNC_PUSH_DD);
                                    }
                                }


            void PreComputationBW() {// Reorganizing nothing, just reset the round and record the workload. 
                const int dev_id = 0;
                const groute::Stream &stream = m_groute_context->CreateStream(dev_id);
                GraphDatum &graph_datum = *m_graph_datum;
                m_running_info.current_round++;
                graph_datum.m_current_round.set_val_H2DAsync(m_running_info.current_round, stream.cuda_stream);                

                stream.Sync();
            }


            void ExecutePolicy_del_con(AlgoVariant *algo_variant) {
                // LOG("------ExecutePolicy_del------\n");
                auto &app_inst = *m_app_inst;
                // auto &vcsr_graph_host = m_vcsr_dev_graph_allocator->HostObject();
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
                Stopwatch sw_execution(true);
		        uint64_t seg_sedge_csr,seg_nedges_csr;
		        index_t seg_snode,seg_enode;
                index_t seg_idx_new;

                m_groute_context->segment_ct = FLAGS_SEGMENT;

                index_t stream_id;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){

    		        seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
    		        seg_enode = m_groute_context->seg_enode[seg_idx];                                    // end node
    		        seg_sedge_csr = m_groute_context->seg_sedge_csr[seg_idx];                            // start edge
    		        seg_nedges_csr = m_groute_context->seg_nedge_csr[seg_idx]; 

    		        stream_id = seg_idx % FLAGS_n_stream;

                    const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                    if(algo_variant[seg_idx] == AlgoVariant::Zero_Copy){
                        m_vcsr_dev_graph_allocator->SwitchZC();
                        RunSyncPushDDB_del(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx,zcflag,
                           vcsr_graph,
                           graph_datum,
                           m_engine_options,
                           stream[stream_id]); 
                    }
                }
                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                        stream[stream_idx].Sync();
                }
               PostComputationBW_del();
               sw_execution.stop();
            //    LOG("iter once for del time: %f ms (excluded)\n", sw_execution.ms());

            }

            void ExecutePolicy_add(AlgoVariant *algo_variant) {
                // LOG("------ExecutePolicy_ADD------\n");
                auto &app_inst = *m_app_inst;
                auto &vcsr_graph_host = m_vcsr_dev_graph_allocator->HostObject();
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
		        uint64_t seg_sedge_csr,seg_nedges_csr;
		        index_t seg_snode,seg_enode;
                index_t seg_idx_new;

                m_groute_context->segment_ct = FLAGS_SEGMENT;
                Stopwatch sw_execution(true);

                index_t stream_id;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
    		        seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
    		        seg_enode = m_groute_context->seg_enode[seg_idx];                                    // end node
    		        seg_sedge_csr = m_groute_context->seg_sedge_csr[seg_idx];                            // start edge
    		        seg_nedges_csr = m_groute_context->seg_nedge_csr[seg_idx]; 

    		        stream_id = seg_idx % FLAGS_n_stream;

                    const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                    if(algo_variant[seg_idx] == AlgoVariant::Zero_Copy){
                        // m_running_info.zerocopy_num++;
                        m_vcsr_dev_graph_allocator->SwitchZC();
                        // m_graph_datum->m_vcsr_edge_weight_datum.SwitchZC();
                        zcflag = true;
                        RunSyncPushDDB_ADD(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx,zcflag,
                           vcsr_graph,
                           graph_datum,
                           m_engine_options,
                           stream[stream_id]); 
                    }
               }
               for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }
                sw_execution.stop();
               LOG("插入刷新缓存 time: %f ms (excluded)\n", sw_execution.ms());
            }

            void ExecutePolicy_Converge(AlgoVariant *algo_variant) {
                // printf("conver iter\n");
                auto &app_inst = *m_app_inst;
                auto &vcsr_graph_host = m_vcsr_dev_graph_allocator->HostObject();
                GraphDatum &graph_datum = *m_graph_datum;
                bool zcflag = true;
                Stopwatch sw_execution(true);
                m_vcsr_dev_graph_allocator->AllocateDevMirror_Edge_Zero();
		        uint64_t seg_sedge_csr,seg_nedges_csr;
		        index_t seg_snode,seg_enode;

                m_groute_context->segment_ct = FLAGS_SEGMENT;
                // vcsr_graph_host
                index_t seg_exc = 0;

                index_t stream_id;
                // LOG("debug 1\n");
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
    		        seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
    		        seg_enode = m_groute_context->seg_enode[seg_idx];                                    // end node
    		        seg_sedge_csr = m_groute_context->seg_sedge_csr[seg_idx];                            // start edge
    		        seg_nedges_csr = m_groute_context->seg_nedge_csr[seg_idx]; 

    		        stream_id = seg_idx % FLAGS_n_stream;

                    const auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                    if(algo_variant[seg_idx] == AlgoVariant::Zero_Copy){
                        m_vcsr_dev_graph_allocator->SwitchZC();
                        zcflag = true;
                        RunSyncPushDDB_Delta(app_inst,seg_snode,seg_enode,seg_sedge_csr,seg_idx,zcflag,
                           vcsr_graph,
                           graph_datum,
                           m_engine_options,
                           stream[stream_id]); 
                    }
               }
               for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                     stream[stream_idx].Sync();
               }

               PostComputationBW();
               
            //    sw_round.stop();

            }

            void PostComputationBW_del() {
                int dev_id = 0;
                const groute::Stream &stream_seg = m_groute_context->CreateStream(dev_id);
                GraphDatum &graph_datum = *m_graph_datum;
                AppImplDeviceObject &app_inst = *m_app_inst;
                m_running_info.current_round = m_graph_datum->m_current_round.get_val_D2H();

                Stopwatch sw_unique(true);

                index_t seg_snode,seg_enode;
                index_t stream_id;
                auto &vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                Stopwatch sw_rebuild(true);
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
			        seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
			        seg_enode = m_groute_context->seg_enode[seg_idx];  
			        RebuildWorklist_del(app_inst,vcsr_graph,
                        graph_datum,
                        stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                }

                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                    stream[stream_idx].Sync();
                }
                sw_rebuild.stop();
                uint64_t del_act_num_node = 0;
                m_running_info.time_overhead_rebuild_worklist += sw_rebuild.ms();
                // for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
                //     stream_id = seg_idx % FLAGS_n_stream;		      

		        //     index_t active_count = graph_datum.m_wl_array_in_seg[seg_idx].GetCount(stream[stream_id]);    
		        //     graph_datum.seg_active_num[seg_idx] = active_count;

		        //     m_running_info.input_active_count_seg[seg_idx] = active_count;

		        //     uint32_t work_size = active_count;
                //     del_act_num_node+=work_size;
                // }
                // std::cout<<"删除活跃顶点 "<<del_act_num_node<<std::endl;
                sw_unique.stop();
                m_running_info.time_overhead_wl_unique += sw_unique.ms();

          }

            void PostComputationBW() {
                // printf("------------PostComputationBW-----------\n");
                int dev_id = 0;
                const groute::Stream &stream_seg = m_groute_context->CreateStream(dev_id);
                GraphDatum &graph_datum = *m_graph_datum;
                AppImplDeviceObject &app_inst = *m_app_inst;
                m_running_info.current_round = m_graph_datum->m_current_round.get_val_D2H();

                Stopwatch sw_unique(true);

                index_t seg_snode,seg_enode;
                index_t stream_id;

                Stopwatch sw_rebuild(true);
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream;
			        seg_snode = m_groute_context->seg_snode[seg_idx];                                    // start node
			        seg_enode = m_groute_context->seg_enode[seg_idx];  
			        RebuildArrayWorklist(app_inst,
                        graph_datum,
                        stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                }

                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                    stream[stream_idx].Sync();
                }
                uint64_t act = 0;
                sw_rebuild.stop();
                m_running_info.time_overhead_rebuild_worklist += sw_rebuild.ms();
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream; 
                    index_t active_count = graph_datum.m_wl_array_in_seg[seg_idx].GetCount(stream[stream_id]);
		            graph_datum.seg_active_num[seg_idx] = active_count;

		            m_running_info.input_active_count_seg[seg_idx] = active_count;

		            uint32_t work_size = active_count;
		            dim3 grid_dims, block_dims;
                    // act += work_size;

                }
                // printf("static num nodes %d\n",act);
                sw_unique.stop();
                m_running_info.time_overhead_wl_unique += sw_unique.ms();

          }

            void count() {
                // printf("------------PostComputationBW-----------\n");
                int dev_id = 0;
                const groute::Stream &stream_seg = m_groute_context->CreateStream(dev_id);
                GraphDatum &graph_datum = *m_graph_datum;
                AppImplDeviceObject &app_inst = *m_app_inst;
                m_running_info.current_round = m_graph_datum->m_current_round.get_val_D2H();

                // Stopwatch sw_unique(true);

                index_t seg_snode,seg_enode;
                index_t stream_id;
                uint64_t act = 0;
                // sw_rebuild.stop();
                // m_running_info.time_overhead_rebuild_worklist += sw_rebuild.ms();
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT ; seg_idx++){
                    stream_id = seg_idx % FLAGS_n_stream; 
                    index_t active_count = graph_datum.m_wl_array_in_seg[seg_idx].GetCount(stream[stream_id]);
		            graph_datum.seg_active_num[seg_idx] = active_count;

		            m_running_info.input_active_count_seg[seg_idx] = active_count;

		            uint32_t work_size = active_count;
		            dim3 grid_dims, block_dims;
                    // act += work_size;

                }
                // printf("static num nodes %d\n",act);
                // sw_unique.stop();
                // m_running_info.time_overhead_wl_unique += sw_unique.ms();

          }
           
           void CombineTask(AlgoVariant *algo_variant) {
                // LOG("------------CombineTask-----------\n");
                int dev_id = 0;
                const groute::Stream &stream_seg = m_groute_context->CreateStream(dev_id);
                GraphDatum &graph_datum = *m_graph_datum;
                AppImplDeviceObject &app_inst = *m_app_inst;
                graph_datum.m_wl_array_in_seg[FLAGS_SEGMENT].ResetAsync(stream_seg.cuda_stream);
                graph_datum.m_wl_array_in_seg[FLAGS_SEGMENT + 1].ResetAsync(stream_seg.cuda_stream);
                stream_seg.Sync();
                Stopwatch sw_unique(true);
                index_t seg_snode,seg_enode;
                index_t stream_id;
                index_t task = 1;// zero:0 exp_filter:1 exp_compaction:2
                bool zc = false;
                bool compaction = false;
                Stopwatch sw_rebuild(true);
                index_t seg_idx_ct = 0;
                for(index_t seg_idx = 0; seg_idx < FLAGS_SEGMENT; seg_idx++){  
                    stream_id = seg_idx % FLAGS_n_stream;
                    seg_snode = m_groute_context->seg_snode[seg_idx];
                    if(algo_variant[seg_idx] == AlgoVariant::Zero_Copy){
                        task = 0;
                        zc = true;
                        while(algo_variant[seg_idx + 1] == AlgoVariant::Zero_Copy && seg_idx < FLAGS_SEGMENT - 1){
                            seg_idx++;
                        }
                    }
                    if(algo_variant[seg_idx] == AlgoVariant::Exp_Compaction){
                        task = 2;
                        compaction = true;
                        // LOG("Compaction\n");
                        while(algo_variant[seg_idx + 1] == AlgoVariant::Exp_Compaction && seg_idx < FLAGS_SEGMENT - 1){
                            seg_idx++;
                        }
                    }
                    seg_enode = m_groute_context->seg_enode[seg_idx];
                    if(task == 0){
                        task = 1;
                        // LOG("zc end %d start %d\n",seg_enode,seg_snode);
                        RebuildArrayWorklist_zero(app_inst,
                            graph_datum,
                            stream[stream_id],seg_snode,seg_enode - seg_snode,FLAGS_SEGMENT);
                    }
                    else if(task == 1)
                    {
                        algo_variant[seg_idx_ct] = AlgoVariant::Exp_Filter;
                        m_groute_context->segment_id_ct[seg_idx_ct++] = seg_idx;
                        // LOG("ef end %d start %d\n",seg_enode,seg_snode);
                        RebuildArrayWorklist(app_inst,
                            graph_datum,
                            stream[stream_id],seg_snode,seg_enode - seg_snode,seg_idx);
                    }
                    else if(task == 2){
                        task = 1;
                        // LOG("ec end %d start %d\n",seg_enode,seg_snode);
                        RebuildArrayWorklist_compaction(app_inst,
                            graph_datum,
                            stream[stream_id],seg_snode,seg_enode - seg_snode,FLAGS_SEGMENT + 1);
                    }
                }
                if(zc){
                    m_groute_context->segment_id_ct[seg_idx_ct] = seg_idx_ct;
                    algo_variant[seg_idx_ct++] = AlgoVariant::Zero_Copy;
                }
                if(compaction){
                    m_groute_context->segment_id_ct[seg_idx_ct] = seg_idx_ct;
                    algo_variant[seg_idx_ct++] = AlgoVariant::Exp_Compaction;
                }

                for(index_t stream_idx = 0; stream_idx < FLAGS_n_stream ; stream_idx++){
                    stream[stream_idx].Sync();
                }
                m_groute_context->segment_ct = seg_idx_ct;
                // printf("seg_idx_ct %d\n======",seg_idx_ct);
                sw_rebuild.stop();
                m_running_info.time_overhead_rebuild_worklist += sw_rebuild.ms();
                for(index_t seg_idx = 0; seg_idx < seg_idx_ct ; seg_idx++){
                    uint32_t seg_idx_new = m_groute_context->segment_id_ct[seg_idx];
                    // printf("seg_idx_new %d",seg_idx_new);
                    stream_id = seg_idx % FLAGS_n_stream;            
                    index_t active_count = graph_datum.m_wl_array_in_seg[seg_idx_new].GetCount(stream[stream_id]);  
                    // printf(": active %d\n",active_count);
                    uint32_t work_size = active_count;
                    dim3 grid_dims, block_dims;

                    if(FLAGS_priority_a == 1){
                        // printf("FLAGS_priority_a ");
                            Stopwatch sw_priority(true);
                            if(algo_variant[seg_idx_new] == AlgoVariant::Zero_Copy){
                                graph_datum.seg_res_num[seg_idx_new] = 1;
                                continue;
                            }
                            if(algo_variant[seg_idx_new] == AlgoVariant::Exp_Compaction){
                                graph_datum.seg_res_num[seg_idx_new] = 0;
                                continue;
                            }
                            graph_datum.m_seg_value.set_val_H2DAsync(0, stream[stream_id].cuda_stream);
                            KernelSizing(grid_dims, block_dims, work_size);
                            kernel::SumResQueue << < grid_dims, block_dims, 0, stream[stream_id].cuda_stream >> >
                                (app_inst,
                                    groute::dev::WorkSourceArray<index_t>(
                                    graph_datum.m_wl_array_in_seg[seg_idx_new].GetDeviceDataPtr(),
                                    work_size),
                                graph_datum.GetValueDeviceObject(),
                                graph_datum.GetBufferDeviceObject(),
                                graph_datum.m_seg_value.dev_ptr);
                                
                            stream[stream_id].Sync();
                            graph_datum.seg_res_num[seg_idx_new] = graph_datum.m_seg_value.get_val_D2H();
                            sw_priority.stop();
                            m_running_info.time_overhead_sample += sw_priority.ms();
                    }

                }
                graph_datum.Compaction_num = 0;
                sw_unique.stop();
                m_running_info.time_overhead_wl_unique += sw_unique.ms();

          }

          void Compaction() {
                int dev_id = 0;
                const groute::Stream &stream_seg = m_groute_context->CreateStream(dev_id);
                GraphDatum &graph_datum = *m_graph_datum;
                AppImplDeviceObject &app_inst = *m_app_inst;
                // auto csr_graph = m_csr_dev_graph_allocator->DeviceObject();
                auto vcsr_graph = m_vcsr_dev_graph_allocator->DeviceObject();
                // auto &csr_graph_host = m_csr_dev_graph_allocator->HostObject();
                auto &vcsr_graph_host = m_vcsr_dev_graph_allocator->HostObject();
                thrust::device_ptr<uint32_t> ptr_labeling(graph_datum.activeNodesLabeling.dev_ptr);
                thrust::device_ptr<uint32_t> ptr_labeling_prefixsum(graph_datum.prefixLabeling.dev_ptr);

                graph_datum.subgraphnodes = thrust::reduce(ptr_labeling, ptr_labeling + graph_datum.nnodes);

                thrust::exclusive_scan(ptr_labeling, ptr_labeling + graph_datum.nnodes, ptr_labeling_prefixsum);

                kernel::makeQueue<<<graph_datum.nnodes/512+1, 512>>>(vcsr_graph.subgraph_activenode, graph_datum.activeNodesLabeling.dev_ptr, graph_datum.prefixLabeling.dev_ptr, graph_datum.nnodes);

                GROUTE_CUDA_CHECK(cudaMemcpy(vcsr_graph_host.subgraph_activenode, vcsr_graph.subgraph_activenode, graph_datum.subgraphnodes*sizeof(uint32_t), cudaMemcpyDeviceToHost));

                thrust::device_ptr<uint32_t> ptr_degrees(graph_datum.activeNodesDegree.dev_ptr);
                thrust::device_ptr<uint32_t> ptr_degrees_prefixsum(graph_datum.prefixSumDegrees.dev_ptr);

                thrust::exclusive_scan(ptr_degrees, ptr_degrees + graph_datum.nnodes, ptr_degrees_prefixsum);

                kernel::makeActiveNodesPointer<<<graph_datum.nnodes/512+1, 512>>>(vcsr_graph.subgraph_rowstart, graph_datum.activeNodesLabeling.dev_ptr, graph_datum.prefixLabeling.dev_ptr, graph_datum.prefixSumDegrees.dev_ptr, graph_datum.nnodes);
                
                GROUTE_CUDA_CHECK(cudaMemcpy(vcsr_graph_host.subgraph_rowstart, vcsr_graph.subgraph_rowstart, graph_datum.subgraphnodes*sizeof(uint32_t), cudaMemcpyDeviceToHost));

                uint32_t numActiveEdges = 0;
                uint32_t endid = vcsr_graph_host.subgraph_activenode[graph_datum.subgraphnodes-1];
                // uint32_t outDegree = vcsr_graph_host.end_edge(endid) - vcsr_graph_host.begin_edge(endid);
                uint64_t outDegree = vcsr_graph_host.sync_vertices_[endid].index;
                if(graph_datum.subgraphnodes > 0)
                    numActiveEdges = vcsr_graph_host.subgraph_rowstart[graph_datum.subgraphnodes-1] + outDegree; 
                
                graph_datum.subgraphedges = numActiveEdges;
                uint32_t last = numActiveEdges;

                GROUTE_CUDA_CHECK(cudaMemcpy(vcsr_graph.subgraph_rowstart + graph_datum.subgraphnodes, &last, sizeof(uint32_t), cudaMemcpyHostToDevice));
    
                GROUTE_CUDA_CHECK(cudaMemcpy(vcsr_graph_host.subgraph_rowstart, vcsr_graph.subgraph_rowstart, (graph_datum.subgraphnodes + 1)*sizeof(uint32_t), cudaMemcpyDeviceToHost));

                uint32_t numThreads = 32;

                if(graph_datum.subgraphnodes < 5000)
                    numThreads = 1;
                std::thread runThreads[numThreads];
          }

      };


  }
}

#endif //HYBRID_FRAMEWORK_H
