// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_SYNC_PUSH_DD_H
#define HYBRID_SYNC_PUSH_DD_H
#define BLOCK_SIZE 1024
#include <utils/cuda_utils.h>
#include <groute/device/cta_scheduler_hybrid.cuh>
#include <framework/variants/push_functor.h>
#include <framework/variants/pull_functor.h>
#include <framework/variants/common.cuh>
#include <framework/common.h>
#include <stdgpu/atomic.cuh>

namespace sepgraph
{
    namespace kernel
    {
        namespace sync_push_dd
        {
            using sepgraph::common::LoadBalancing;
            
            template<typename TAppInst,
                    typename WorkSource,
                    typename WorkTarget,
                    typename PMAGraph,
                    template<typename> class GraphDatum,
                    typename TValue,
                    typename TBuffer,
                    typename TWeight>
            __forceinline__ __device__
            void Relax(TAppInst app_inst,
		       index_t seg_snode,
			   index_t seg_enode,
			   uint64_t seg_sedge_csr,
		       bool zcflag,
                       WorkSource work_source,
                       WorkTarget work_target_low,
                       WorkTarget work_target_high,
                       TBuffer current_priority,
                       PMAGraph csr_graph,
                       GraphDatum<TValue> node_value_datum,
                       GraphDatum<TBuffer> node_buffer_datum,
                       GraphDatum<TWeight> edge_weight_datum)
            {
                uint32_t tid = TID_1D;
                uint32_t nthreads = TOTAL_THREADS_1D;
                uint32_t work_size = work_source.get_size();
		
                PushFunctor<TAppInst, PMAGraph, GraphDatum, TBuffer, TWeight>
                        push_functor(app_inst,
                                     work_target_low,
                                     work_target_high,
                                     current_priority,
                                     csr_graph,
                                     node_buffer_datum,
                                     edge_weight_datum);
		
		 
		
               /*for (int i = 0 + tid; i < work_size; i += nthreads)
                {
                    index_t node = work_source.get_work(i);
		            if(node >= seg_snode && node < seg_enode){
		                auto pair = app_inst.CombineValueBuffer(node,
                                                            node_value_datum.get_item_ptr(node),
                                                            node_buffer_datum.get_item_ptr(node));
				 printf("node: %d buffer_to_push: %d\n", node, pair.first);
		                if (pair.second)
                        {
                            Payload<TBuffer> payload;
                            payload.m_src = node;
                            payload.m_buffer_to_push = pair.first;
			    printf("node: %d buffer_to_push: %d\n", node, pair.first);
			                if(zcflag == false){
                                for (index_t edge = csr_graph.begin_edge(node) - seg_sedge_csr, end_edge = csr_graph.end_edge(node) - seg_sedge_csr; edge < end_edge; edge++){
                                    if (!push_functor(edge, payload)){
                                        break;
                                    }
                                }
			                }else{
			                    for (index_t edge = csr_graph.begin_edge(node), end_edge = csr_graph.end_edge(node); edge < end_edge; edge++){
                                    if (!push_functor(edge, payload)){
                                        break;
                                    }
                                }
			                }
                        }
		            }
		        }*/
            }
           template<LoadBalancing LB,
                    bool enable_priority,
                    typename TAppInst,
                    typename WorkSource,
                    typename PMAGraph,
                    typename BufferVec,
                    typename TValue,
                    typename TBuffer>
            __forceinline__ __device__
            void RelaxCTADBFlush(TAppInst app_inst,
                            index_t seg_snode,
                            index_t seg_enode,
                            uint64_t seg_sedge_csr,
                            bool zcflag,
                            WorkSource work_source,
                            // const PMAGraph vcsr_graph,
                            PMAGraph vcsr_graph,
                            BufferVec *buffer,
                            BufferVec *buffer_l2,
                            BufferVec *buffer_l3,
                            uint64_t *cache_size,
                            TValue* node_value_datum,
                            TValue* node_parent_datum,
                            TBuffer* node_buffer_datum,
                            TBuffer current_priority,
                            BitmapDeviceObject out_active,
                            BitmapDeviceObject in_active)
            {
                const uint32_t tid = TID_1D;
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                int first = 1;
                PushFunctorDBCacheFlush<TAppInst, PMAGraph, BufferVec ,TBuffer>
                        push_functor_l1(app_inst, vcsr_graph, buffer, buffer_l3,cache_size, first,out_active);

                for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
                {
                    groute::dev::np_local<Payload<TBuffer>> np_local_ca = {0, 0};

                    if (tid < work_size)
                    {
                        //遍历cache_l1和cache_l2，对于 virtual_degree > 0 将数据合并到一个里面
                        const index_t node = work_source.get_work(tid);                          
                        Payload<TBuffer> payload;
                        payload.m_src = node;
                    
                        np_local_ca.start = vcsr_graph.vertices_[node].virtual_start;
                        np_local_ca.size = vcsr_graph.vertices_[node].virtual_degree;
                        np_local_ca.meta_data = payload;
                        index_t node_index = atomicAdd(vcsr_graph.river_global, (index_t)(np_local_ca.size));
                        vcsr_graph.vertices_[node].third_start = (uint64_t)node_index;
            
                    }
                    switch (LB)
                    {
                        case LoadBalancing::COARSE_GRAINED:
                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                schedule(np_local_ca, push_functor_l1,zcflag);
                            break;
                        case LoadBalancing::FINE_GRAINED:
                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                schedule(np_local_ca, push_functor_l1,zcflag);
                            break;
                        case LoadBalancing::HYBRID:
                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                schedule(np_local_ca, push_functor_l1,zcflag);
                            break;
                        default:
                            assert(false);
                    }

                }
            }

           template<LoadBalancing LB,
                    bool enable_priority,
                    typename TAppInst,
                    typename WorkSource,
                    typename PMAGraph,
                    typename Hot,
                    typename BufferVec,
                    typename TValue,
                    typename TBuffer>
            __forceinline__ __device__
            void RelaxCTADB(TAppInst app_inst,
                            index_t seg_snode,
                            index_t seg_enode,
                            uint64_t seg_sedge_csr,
                            bool zcflag,
                            WorkSource work_source,
                            PMAGraph vcsr_graph,
                            Hot hotness,
                            BufferVec *buffer,
                            BufferVec *buffer_l2,
                            uint64_t *count_gpu,
                         uint64_t *total_act_d,
                            uint64_t *cache_size,
                            TValue *node_parent_datum,
                            TValue *node_value_datum,
                            TBuffer * node_buffer_datum,
                            TBuffer current_priority,
                            BitmapDeviceObject out_active,
                            BitmapDeviceObject in_active)
            {
                const uint32_t tid = TID_1D;
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctorDB<TAppInst, PMAGraph, BufferVec, TValue, TBuffer>
                        push_functor(app_inst, vcsr_graph, buffer,cache_size, node_parent_datum, node_buffer_datum, out_active);
                PushFunctorDBCachel2<TAppInst, PMAGraph, BufferVec, TValue, TBuffer>
                        push_functor_l1(app_inst, vcsr_graph, buffer, node_parent_datum, node_buffer_datum, out_active);
                        for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
                        {
                            groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};
                            groute::dev::np_local<Payload<TBuffer>> np_local_ca = {0, 0};
                            if (tid < work_size)
                            {
                                const index_t node = work_source.get_work(tid);
                                const auto pair = app_inst.CombineValueBuffer(node,
                                &node_value_datum[node],&node_buffer_datum[node]);
                                // Value changed means validate combine, we need push the buffer to neighbors
                                if (pair.second)
                                {                                       
                                    Payload<TBuffer> payload;
                                    payload.m_src = node;
                                    payload.m_buffer_to_push = pair.first;
                                    if(vcsr_graph.vertices_[node].cache){
                                        vcsr_graph.vertices_[node].hotness[0] +=1;                           
                                        np_local_ca.start = vcsr_graph.vertices_[node].virtual_start;
                                        np_local_ca.size = vcsr_graph.vertices_[node].virtual_degree;
                                        np_local_ca.meta_data  = payload;
                                        // total_act_d[node]++;
                                        // total_act_d[node]+=np_local_ca.size;
                                    }else{
                                        np_local.start = vcsr_graph.begin_edge(node);
                                        vcsr_graph.vertices_[node].hotness[0]++;
                                        np_local.size =vcsr_graph.sync_vertices_[node].degree;
                                        np_local.meta_data = payload;
                                        // total_act_d[node]++;
                                        // total_act_d[node]+=np_local.size;
                                        // count_gpu[node]++;
                                    }

                                }
                                
                            } 
                            switch (LB)
                            {
                                case LoadBalancing::COARSE_GRAINED:
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                        schedule(np_local_ca, push_functor_l1,zcflag);
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                        schedule(np_local, push_functor,zcflag);
                                    break;
                                case LoadBalancing::FINE_GRAINED:
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                        schedule(np_local_ca, push_functor_l1,zcflag);
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                                        schedule(np_local, push_functor,zcflag);
                                    break;
                                case LoadBalancing::HYBRID:
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                        schedule(np_local_ca, push_functor_l1,zcflag);
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                                        schedule(np_local, push_functor,zcflag);
                                    break;
                                default:
                                    assert(false);
                            }

                        }
            }

            //push function for incremental compuatation (including cache policy)
           template<LoadBalancing LB,
                    bool enable_priority,
                    typename TAppInst,
                    typename WorkSource,
                    typename PMAGraph,
                    typename BufferVec,
                    typename TValue,
                    typename TBuffer>
            __forceinline__ __device__
            void RelaxCTADBMFC(TAppInst app_inst,
                            index_t seg_snode,
                            index_t seg_enode,
                            uint64_t seg_sedge_csr,
                            bool zcflag,
                            WorkSource work_source,
                            PMAGraph vcsr_graph,
                            BufferVec *buffer,
                            BufferVec *buffer_l2,
                            uint64_t *count_gpu,
                            uint64_t *transfer,
                            uint64_t *cache_size,
                            TValue *node_parent_datum,
                            TValue *node_value_datum,
                            TBuffer * node_buffer_datum,
                            TBuffer current_priority,
                            BitmapDeviceObject out_active,
                            BitmapDeviceObject in_active)
            {
                const uint32_t tid = TID_1D;
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctorDB<TAppInst, PMAGraph, BufferVec, TValue, TBuffer>
                        push_functor(app_inst, vcsr_graph, buffer,cache_size, node_parent_datum, node_buffer_datum, out_active);
                PushFunctorDBCachel2<TAppInst, PMAGraph, BufferVec, TValue, TBuffer>
                        push_functor_l1(app_inst, vcsr_graph, buffer, node_parent_datum, node_buffer_datum, out_active);
                        for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
                        {
                            groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};
                            groute::dev::np_local<Payload<TBuffer>> np_local_ca = {0, 0};
                            if (tid < work_size)
                            {
                                const index_t node = work_source.get_work(tid);
                                const auto pair = app_inst.CombineValueBuffer(node,
                                &node_value_datum[node],&node_buffer_datum[node]);
                                // Value changed means validate combine, we need push the buffer to neighbors
                                if (pair.second)
                                {                                       
                                    Payload<TBuffer> payload;
                                    payload.m_src = node;
                                    payload.m_buffer_to_push = pair.first;
                                    if(vcsr_graph.vertices_[node].cache){     
                                        np_local_ca.start = vcsr_graph.vertices_[node].virtual_start;
                                        np_local_ca.size = vcsr_graph.vertices_[node].virtual_degree;
                                        np_local_ca.meta_data  = payload;
                                        vcsr_graph.vertices_[node].hotness[0]+=1;
                                        transfer[node] += np_local_ca.size;
                                    }else{
                                        //cache miss
                                        np_local.start = vcsr_graph.begin_edge(node);
                                        np_local.size =vcsr_graph.sync_vertices_[node].degree;
                                        vcsr_graph.vertices_[node].hotness[0]+=1;
                                        np_local.meta_data = payload;
                                        transfer[node] +=np_local.size;
                                    }
                                }
                                
                            } 
                            switch (LB)
                            {
                                case LoadBalancing::COARSE_GRAINED:

                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                        schedule(np_local_ca, push_functor_l1,zcflag);
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                        schedule(np_local, push_functor,zcflag);
                                    break;
                                case LoadBalancing::FINE_GRAINED:
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                                        schedule(np_local_ca, push_functor_l1,zcflag);
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                                        schedule(np_local, push_functor,zcflag);
                                    break;
                                case LoadBalancing::HYBRID:
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                                        schedule(np_local_ca, push_functor_l1,zcflag);
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                                        schedule(np_local, push_functor,zcflag);
                                    break;
                                default:
                                    assert(false);
                            }

                        }
            }
           template<LoadBalancing LB,
                    bool enable_priority,
                    typename TAppInst,
                    typename WorkSource,
                    typename PMAGraph,
                    typename BufferVec,
                    typename TValue,
                    typename TBuffer>
            __forceinline__ __device__
            void RelaxCTADBAmend(TAppInst app_inst,
                            index_t seg_snode,
                            index_t seg_enode,
                            uint64_t seg_sedge_csr,
                            bool zcflag,
                            WorkSource work_source,
                            // const PMAGraph vcsr_graph,
                            PMAGraph vcsr_graph,
                            int* type_device,
                            BufferVec *buffer,
                            BufferVec *buffer_l2,
                            uint64_t *total_act_d,
                            uint64_t *cache_size,
                            TValue *node_parent_datum,
                            TValue *node_value_datum,
                            TBuffer * node_buffer_datum,
                            // TBuffer * node_tmp_buffer_datum,
                            // TValue *node_level_datum,
                            // GraphDatum<TWeight> edge_weight_datum,
                            TBuffer current_priority,
                            BitmapDeviceObject out_active,
                            BitmapDeviceObject in_active)
            {
                const uint32_t tid = TID_1D;
                // if(tid==0)printf("RelaxCTADB\n");
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctorDB<TAppInst, PMAGraph, BufferVec, TValue, TBuffer>
                        push_functor(app_inst, vcsr_graph, buffer,cache_size, node_parent_datum, node_buffer_datum, out_active);
                PushFunctorDBCachel2<TAppInst, PMAGraph, BufferVec, TValue, TBuffer>
                        push_functor_l1(app_inst, vcsr_graph, buffer, node_parent_datum, node_buffer_datum, out_active);
                        for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
                        {
                            groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};
                            groute::dev::np_local<Payload<TBuffer>> np_local_ca = {0, 0};
                            if (tid < work_size)
                            {
                                const index_t node = work_source.get_work(tid);
                                const auto pair = app_inst.CombineValueBufferAmend(node,type_device,
                                &node_value_datum[node],&node_buffer_datum[node]);
                                // const auto pair = app_inst.CombineValueBuffer(node,&node_value_datum[node],&node_buffer_datum[node]);
                                // Value changed means validate combine, we need push the buffer to neighbors
                                // bool correct = vcsr_graph.vertices_[node].virtual_degree ==vcsr_graph.sync_vertices_[node].degree ? true :false;
                                assert(vcsr_graph.vertices_[node].virtual_degree ==  vcsr_graph.sync_vertices_[node].degree ) ;
                                assert((vcsr_graph.vertices_[node].virtual_start + vcsr_graph.vertices_[node].virtual_degree)  < (*cache_size)) ;
                                 
                                if (pair.second)
                                {                                       
                                    Payload<TBuffer> payload;
                                    payload.m_src = node;
                                    payload.m_buffer_to_push = pair.first;
                                    if(vcsr_graph.vertices_[node].cache){
                                        np_local_ca.start = vcsr_graph.vertices_[node].virtual_start;
                                        np_local_ca.size = vcsr_graph.vertices_[node].virtual_degree;
                                        np_local_ca.meta_data  = payload;
                                        // total_act_d[node]+=np_local_ca.size;
                                    }else{
                                        np_local.start = vcsr_graph.begin_edge(node);
                                        np_local.size = vcsr_graph.sync_vertices_[node].degree;
                                        np_local.meta_data = payload;
                                        // total_act_d[node]+=np_local.size;
                                    }

                                }
                                
                            } 
                            switch (LB)
                            {
                                case LoadBalancing::COARSE_GRAINED:
                                        // printf("Use COARSE_GRAINED\n" );
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                        schedule(np_local_ca, push_functor_l1,zcflag);
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                        schedule(np_local, push_functor,zcflag);
                                    break;
                                case LoadBalancing::FINE_GRAINED:
                                // printf("Use FINE_GRAINED\n" );
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                                        schedule(np_local_ca, push_functor_l1,zcflag);
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                                        schedule(np_local, push_functor,zcflag);
                                    break;
                                case LoadBalancing::HYBRID:
                                // printf("Use HYBRID\n" );
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                                        schedule(np_local_ca, push_functor_l1,zcflag);
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                                        schedule(np_local, push_functor,zcflag);
                                    break;
                                default:
                                    assert(false);
                            }

                        }
            }

           template<LoadBalancing LB,
                    bool enable_priority,
                    typename TAppInst,
                    typename WorkSource,
                    typename PMAGraph,
                    typename BufferVec,
                    typename TValue,
                    typename TBuffer>
            __forceinline__ __device__
            void RelaxCTADBCache(TAppInst app_inst,
                            index_t seg_snode,
                            index_t seg_enode,
                            uint64_t seg_sedge_csr,
                            bool zcflag,
                            WorkSource work_source,
                            // const PMAGraph vcsr_graph,
                            PMAGraph vcsr_graph,
                            int* type_device,
                            BufferVec *buffer,
                            BufferVec *buffer_l2,
                            uint64_t *cache_size,
                            TValue *node_parent_datum,
                            TValue *node_value_datum,
                            TBuffer * node_buffer_datum,
                            // TBuffer * node_tmp_buffer_datum,
                            // TValue *node_level_datum,
                            // GraphDatum<TWeight> edge_weight_datum,
                            TBuffer current_priority,
                            BitmapDeviceObject out_active,
                            BitmapDeviceObject in_active)
            {
                const uint32_t tid = TID_1D;
                // if(tid==0)printf("RelaxCTADB\n");
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctorDBCachel2<TAppInst, PMAGraph, BufferVec, TValue, TBuffer>
                        push_functor_l1(app_inst, vcsr_graph, buffer, node_parent_datum, node_buffer_datum, out_active);
                        for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
                        {
                            groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};
                            // groute::dev::np_local<Payload<TBuffer>> np_local_ca = {0, 0};
                            if (tid < work_size)
                            {
                                const index_t node = work_source.get_work(tid);
                                const auto pair = app_inst.CombineValueBufferAmend(node,type_device,
                                &node_value_datum[node],&node_buffer_datum[node]);
                                // Value changed means validate combine, we need push the buffer to neighbors
                                if (pair.second)
                                {                                       
                                    Payload<TBuffer> payload;
                                    payload.m_src = node;
                                    payload.m_buffer_to_push = pair.first;
                                    if(vcsr_graph.vertices_[node].cache){
                                        np_local.start = vcsr_graph.vertices_[node].virtual_start;
                                        np_local.size = vcsr_graph.vertices_[node].virtual_degree;
                                        np_local.meta_data = payload;
                                    }

                                }
                                
                            } 
                            switch (LB)
                            {
                                case LoadBalancing::COARSE_GRAINED:

                                        // groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                        // schedule(np_local_ca, push_functor_l1,zcflag);
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                        schedule(np_local, push_functor_l1,zcflag);
                                    break;
                                case LoadBalancing::FINE_GRAINED:
                                        // groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                                        // schedule(np_local_ca, push_functor_l1,zcflag);
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                                        schedule(np_local, push_functor_l1,zcflag);
                                    break;
                                case LoadBalancing::HYBRID:
                                        // groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                                        // schedule(np_local_ca, push_functor_l1,zcflag);
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                                        schedule(np_local, push_functor_l1,zcflag);
                                    break;
                                default:
                                    assert(false);
                            }

                        }
            }

            template<LoadBalancing LB,
                    bool enable_priority,
                    typename TAppInst,
                    typename WorkSource,
                    typename PMAGraph,
                    template<typename> class GraphDatum,
                    typename TValue,
                    typename TBuffer,
                    typename TWeight>
            __forceinline__ __device__
            void RelaxCTADB_COM(TAppInst app_inst,
                                WorkSource work_source,
                                const PMAGraph csr_graph,
                                TValue *node_value_datum,
                                TBuffer *node_buffer_datum,
                                TValue *node_parent_datum,
                                GraphDatum<TWeight> edge_weight_datum,
                                TBuffer current_priority,
                                BitmapDeviceObject out_active,
                                BitmapDeviceObject in_active)
            {
                const uint32_t tid = TID_1D;
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctorDB_COM<TAppInst, PMAGraph, GraphDatum, TValue, TBuffer, TWeight>
                        push_functor(app_inst, csr_graph, node_parent_datum,node_buffer_datum, edge_weight_datum,out_active);

                        for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
                       {
                            groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};
        
                            if (tid < work_size)
                            {
                                    const index_t node = csr_graph.subgraph_activenode[tid];
                                    //printf("node:%d\n",node);
                                    const auto pair = app_inst.CombineValueBuffer(node,
                                                                                  &node_value_datum[node],
                                                                                  &node_buffer_datum[node]);
                                    //out_active.set_bit_atomic(node);
                                    // Value changed means validate combine, we need push the buffer to neighbors
                                    if (pair.second)
                                    {       
                                        np_local.start = csr_graph.subgraph_rowstart[tid];
                                        np_local.size = csr_graph.subgraph_rowstart[tid + 1] - np_local.start; // out-degree
                                        Payload<TBuffer> payload;
                                        payload.m_src = node;
                                        payload.m_buffer_to_push = pair.first;
                                        np_local.meta_data = payload;
                                        
                                        //printf("%d %d %d\n",node, np_local.start, np_local.size);
                                        // if(node==10767||node==785951||node==828471||node==851670)
                                        //     for(int j=np_local.start;j<np_local.start+np_local.size;j++){
                                        //         printf("##%d %d\n",node,csr_graph.edge_dest(j));
                                        //     }
                                    }
                                
                
                             }
        
                            switch (LB)
                            {
                                case LoadBalancing::COARSE_GRAINED:
                                    groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                    schedule(np_local, push_functor,false);
                                    break;
                                case LoadBalancing::FINE_GRAINED:
                                    groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                                    schedule(np_local, push_functor,false);
                                    break;
                                case LoadBalancing::HYBRID:
                                    groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                                    schedule(np_local, push_functor,false);
                                    break;
                                default:
                                    assert(false);
                            }
                        }
            }

           template<LoadBalancing LB,
                    bool enable_priority,
                    typename TAppInst,
                    typename WorkSource,
                    typename PMAGraph,
                    typename BufferVec,
                    typename TValue,
                    typename TBuffer>
            __forceinline__ __device__
            void RelaxCTADBAmend_cache(TAppInst app_inst,
                            index_t seg_snode,
                            index_t seg_enode,
                            uint64_t seg_sedge_csr,
                            bool zcflag,
                            WorkSource work_source,
                            // const PMAGraph vcsr_graph,
                            PMAGraph vcsr_graph,
                            BufferVec *buffer,
                            BufferVec *buffer_l2,
                            uint64_t *cache_size,
                            TValue *node_parent_datum,
                            TValue *node_value_datum,
                            TBuffer * node_buffer_datum,
                            TBuffer current_priority,
                            BitmapDeviceObject out_active,
                            BitmapDeviceObject in_active)
            {
                const uint32_t tid = TID_1D;
                // if(tid==0)printf("RelaxCTADBAmend %d\n",*type_device);
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctorDBAmend_cache<TAppInst, PMAGraph, BufferVec, TValue, TBuffer>
                        push_functor(app_inst, vcsr_graph, buffer,cache_size, node_parent_datum, node_buffer_datum, out_active);
                        for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
                        {
                            groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};
                            // groute::dev::np_local<Payload<TBuffer>> np_local_secondary = {0, 0};
                            if (tid < work_size)
                            {
                                const index_t node = work_source.get_work(tid);
                                // const auto pair = app_inst.CombineValueBufferAmend(node,type_device,
                                // &node_value_datum[node],&node_buffer_datum[node]);
                                // // Value changed means validate combine, we need push the buffer to neighbors
                                // if (pair.second)
                                // {                                       
                                    Payload<TBuffer> payload;
                                    payload.m_src = node;
                                    // uint64_t cache_size_river = *cache_size;
                                    // uint64_t river = *vcsr_graph.river + vcsr_graph.sync_vertices_[node].degree;
                                    // if(river < cache_size_river){
                                        np_local.start = vcsr_graph.begin_edge(node);
                                        np_local.size = vcsr_graph.sync_vertices_[node].degree;
                                        // index_t degree = vcsr_graph.sync_vertices_[node].degree;
                                        index_t node_index = atomicAdd(vcsr_graph.river, vcsr_graph.sync_vertices_[node].degree);
                                        
                                        vcsr_graph.vertices_[node].cache = true;
                                        vcsr_graph.vertices_[node].delta= false;
                                        vcsr_graph.vertices_[node].virtual_start = (uint64_t)node_index;
                                        np_local.meta_data = payload;
                                    // }
                                    // np_local.start = vcsr_graph.begin_edge(node);
                                    // np_local.size = vcsr_graph.sync_vertices_[node].degree;
                                    // // index_t degree = vcsr_graph.sync_vertices_[node].degree;
                                    // index_t node_index = atomicAdd(vcsr_graph.river, vcsr_graph.sync_vertices_[node].degree);
                                    
                                    // vcsr_graph.vertices_[node].cache = true;
                                    // vcsr_graph.vertices_[node].delta= false;
                                    // vcsr_graph.vertices_[node].virtual_start = (uint64_t)node_index;
                                    // np_local.meta_data = payload;
                                // }
                                
                            }
                            switch (LB)
                            {
                                case LoadBalancing::COARSE_GRAINED:
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                        schedule(np_local, push_functor,zcflag);
                                    break;
                                case LoadBalancing::FINE_GRAINED:
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                                        schedule(np_local, push_functor,zcflag);
                                    break;
                                case LoadBalancing::HYBRID:
                                        groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                                        schedule(np_local, push_functor,zcflag);
                                    break;
                                default:
                                    assert(false);
                            }

                        }
            }

            template<LoadBalancing LB,
                    typename TAppInst,
                    typename WorkSource,
                    typename WorkTarget,
                    typename PMAGraph,
                    template<typename> class GraphDatum,
                    typename TValue,
                    typename TBuffer,
                    typename TWeight>
            __forceinline__ __device__
            void RelaxCTA(TAppInst app_inst,
                          index_t seg_snode,
                          index_t seg_enode,
                          index_t seg_sedge_csr,
                          bool zcflag,
                          WorkSource work_source,
                          WorkTarget work_target_low,
                          WorkTarget work_target_high,
                          TBuffer current_priority,
                          PMAGraph csr_graph,
                          GraphDatum<TValue> node_value_datum,
                          GraphDatum<TBuffer> node_buffer_datum,
                          GraphDatum<TWeight> edge_weight_datum)
            {
                const uint32_t tid = TID_1D;
                // if(tid==0) printf("PUSHDD\n");
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctor<TAppInst, PMAGraph, GraphDatum, TBuffer, TWeight>
                        push_functor(app_inst, work_target_low, work_target_high, current_priority,
                                     csr_graph, node_buffer_datum, edge_weight_datum);

               /* for (int i = 0 + tid; i < work_size_rup; i += nthreads)
                {
                    groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};
                    index_t node_to_process=seg_enode+1;

                    if (i < work_size)
                    {
                        const index_t node = work_source.get_work(i);
                        node_to_process=node;
                        auto pair = app_inst.CombineValueBuffer(node,
                                                                node_value_datum.get_item_ptr(node),
                                                                node_buffer_datum.get_item_ptr(node));

                        if (pair.second)
                        {
                            if(zcflag){
                                np_local.start = csr_graph.begin_edge(node);
                                np_local.size = csr_graph.end_edge(node) - np_local.start; // out-degree
                            }else{
                                np_local.start = csr_graph.begin_edge(node) - seg_sedge_csr;
                                np_local.size = csr_graph.end_edge(node) - seg_sedge_csr - np_local.start; // out-degree
                            }
                            Payload<TBuffer> payload;
                            payload.m_src = node;
                            payload.m_buffer_to_push = pair.first;
                            np_local.meta_data = payload;
                        }
                    }
                    
                    if(np_local.meta_data.m_src >= seg_snode && np_local.meta_data.m_src < seg_enode){
                        switch (LB)
                        {
                            case LoadBalancing::COARSE_GRAINED:
                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                schedule(np_local, push_functor);
                                break;
                            case LoadBalancing::FINE_GRAINED:
                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                                schedule(np_local, push_functor);
                                break;
                            case LoadBalancing::HYBRID:
                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                                schedule(np_local, push_functor);
                                break;
                            default:
                                assert(false);
                        }
                    }
                }*/
            }

        template<typename TAppInst,
            typename WorkSource,
            typename WorkTarget,
            typename PMAGraph,
            template<typename> class GraphDatum,
            typename TValue,
            typename TBuffer,
            typename TWeight>
    __forceinline__ __device__
    void Relax_ZC(TAppInst app_inst,
       index_t seg_snode,
       index_t seg_enode,
       uint64_t seg_sedge_csr,
       bool zcflag,
               WorkSource work_source,
               WorkTarget work_target_low,
               WorkTarget work_target_high,
               TBuffer current_priority,
               PMAGraph csr_graph,
               GraphDatum<TValue> node_value_datum,
               GraphDatum<TBuffer> node_buffer_datum,
               GraphDatum<TWeight> edge_weight_datum)
    {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;
        uint32_t work_size = work_source.get_size();

        PushFunctor<TAppInst, PMAGraph, GraphDatum, TBuffer, TWeight>
                push_functor(app_inst,
                             work_target_low,
                             work_target_high,
                             current_priority,
                             csr_graph,
                             node_buffer_datum,
                             edge_weight_datum);

 

        for (int i = 0 + tid; i < work_size; i += nthreads)
        {
            index_t node = work_source.get_work(i);
            if(node >= seg_snode && node < seg_enode){
                auto pair = app_inst.CombineValueBuffer(node,
                                                    node_value_datum.get_item_ptr(node),
                                                    node_buffer_datum.get_item_ptr(node));
                if (pair.second)
                {
                    Payload<TBuffer> payload;
                    payload.m_src = node;
                    payload.m_buffer_to_push = pair.first;
                    if(zcflag == false){
                        for (index_t edge = csr_graph.begin_edge(node) - seg_sedge_csr, end_edge = csr_graph.end_edge(node) - seg_sedge_csr; edge < end_edge; edge++){
                            if (!push_functor(edge, payload)){
                                break;
                            }
                        }
                    }else{
                        for (index_t edge = csr_graph.begin_edge(node), end_edge = csr_graph.end_edge(node); edge < end_edge; edge++){
                            if (!push_functor(edge, payload)){
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

        template<typename TAppInst,
            typename WorkSource,
            typename WorkTarget,
            typename PMAGraph,
            template<typename> class GraphDatum,
            typename TValue,
            typename TBuffer,
            typename TWeight>
    __forceinline__ __device__
    void Relax_segment(TAppInst app_inst,
       index_t seg_snode,
       index_t seg_enode,
       uint64_t seg_sedge_csr,
       bool zcflag,
               WorkSource work_source,
               WorkTarget work_target_low,
               WorkTarget work_target_high,
               TBuffer current_priority,
               PMAGraph csr_graph,
               GraphDatum<TValue> node_value_datum,
               GraphDatum<TBuffer> node_buffer_datum,
               GraphDatum<TWeight> edge_weight_datum)
    {
        uint32_t tid = TID_1D;
        uint32_t nthreads = TOTAL_THREADS_1D;
        uint32_t work_size = work_source.get_size();

        PushFunctor<TAppInst, PMAGraph, GraphDatum, TBuffer, TWeight>
                push_functor(app_inst,
                             work_target_low,
                             work_target_high,
                             current_priority,
                             csr_graph,
                             node_buffer_datum,
                             edge_weight_datum);

 

        for (int i = 0 + tid; i < work_size; i += nthreads)
        {
            index_t node = work_source.get_work(i);
            if(node >= seg_snode && node < seg_enode){
                auto pair = app_inst.CombineValueBuffer(node,
                                                    node_value_datum.get_item_ptr(node),
                                                    node_buffer_datum.get_item_ptr(node));
                if (pair.second)
                {
                    Payload<TBuffer> payload;
                    payload.m_src = node;
                    payload.m_buffer_to_push = pair.first;
                    for (index_t edge = csr_graph.begin_edge(node) - seg_sedge_csr, end_edge = csr_graph.end_edge(node) - seg_sedge_csr; edge < end_edge; edge++){
                        if (!push_functor(edge, payload)){
                            break;
                        }
                    }
                }
            }
        }
    }


            template<LoadBalancing LB,
                    typename TAppInst,
                    typename WorkSource,
                    typename WorkTarget,
                    typename PMAGraph,
                    template<typename> class GraphDatum,
                    typename TValue,
                    typename TBuffer,
                    typename TWeight>
            __forceinline__ __device__
            void RelaxCTA_legacy(TAppInst app_inst,
                          WorkSource work_source,
                          WorkTarget work_target_low,
                          WorkTarget work_target_high,
                          TBuffer current_priority,
                          PMAGraph csr_graph,
                          GraphDatum<TValue> node_value_datum,
                          GraphDatum<TBuffer> node_buffer_datum,
                          GraphDatum<TWeight> edge_weight_datum)
            {
                const uint32_t tid = TID_1D;
                if(tid==0) printf("PUSHDD\n");
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctor<TAppInst, PMAGraph, GraphDatum, TBuffer, TWeight>
                        push_functor(app_inst, work_target_low, work_target_high, current_priority,
                                     csr_graph, node_buffer_datum, edge_weight_datum);

                for (int i = 0 + tid; i < work_size_rup; i += nthreads)
                {
                    groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};

                    if (i < work_size)
                    {
                        const index_t node = work_source.get_work(i);
                        auto pair = app_inst.CombineValueBuffer(node,
                                                                node_value_datum.get_item_ptr(node),
                                                                node_buffer_datum.get_item_ptr(node));

                        if (pair.second)
                        {
                            np_local.start = csr_graph.begin_edge(node);
                            np_local.size = csr_graph.end_edge(node) - np_local.start; // out-degree
                            Payload<TBuffer> payload;
                            payload.m_src = node;
                            payload.m_buffer_to_push = pair.first;
                            np_local.meta_data = payload;
                        }
                    }

                    switch (LB)
                    {
                        case LoadBalancing::COARSE_GRAINED:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                            schedule(np_local, push_functor);
                            break;
                        case LoadBalancing::FINE_GRAINED:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                            schedule(np_local, push_functor);
                            break;
                        case LoadBalancing::HYBRID:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                            schedule(np_local, push_functor);
                            break;
                        default:
                            assert(false);
                    }
                }
            }

            template<LoadBalancing LB,
                    typename TAppInst,
                    typename WorkSource,
                    typename WorkTarget,
                    typename PMAGraph,
                    template<typename> class GraphDatum,
                    typename TValue,
                    typename TBuffer,
                    typename TWeight>
            __forceinline__ __device__
            void RelaxCTA_ZC(TAppInst app_inst,
                          index_t seg_snode,
                          index_t seg_enode,
                          index_t seg_sedge_csr,
                          WorkSource work_source,
                          WorkTarget work_target_low,
                          WorkTarget work_target_high,
                          TBuffer current_priority,
                          PMAGraph csr_graph,
                          GraphDatum<TValue> node_value_datum,
                          GraphDatum<TBuffer> node_buffer_datum,
                          GraphDatum<TWeight> edge_weight_datum)
            {
                const uint32_t tid = TID_1D;
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctor<TAppInst, PMAGraph, GraphDatum, TBuffer, TWeight>
                        push_functor(app_inst, work_target_low, work_target_high, current_priority,
                                     csr_graph, node_buffer_datum, edge_weight_datum);

                for (int i = 0 + tid; i < work_size_rup; i += nthreads)
                {
                    groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};

                    if (i < work_size)
                    {
                        const index_t node = work_source.get_work(i);
                        auto pair = app_inst.CombineValueBuffer(node,
                                                                node_value_datum.get_item_ptr(node),
                                                                node_buffer_datum.get_item_ptr(node));

                        if (pair.second)
                        {
                            np_local.start = csr_graph.begin_edge(node);
                            np_local.size = csr_graph.end_edge(node) - np_local.start; // out-degree
                            Payload<TBuffer> payload;
                            payload.m_src = node;
                            payload.m_buffer_to_push = pair.first;
                            np_local.meta_data = payload;
                        }
                    }

                    switch (LB)
                    {
                        case LoadBalancing::COARSE_GRAINED:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                            schedule(np_local, push_functor);
                            break;
                        case LoadBalancing::FINE_GRAINED:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                            schedule(np_local, push_functor);
                            break;
                        case LoadBalancing::HYBRID:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                            schedule(np_local, push_functor);
                            break;
                        default:
                            assert(false);
                    }
                }
            }

            template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename WorkTarget,
                typename PMAGraph,
                template<typename> class GraphDatum,
                typename TValue,
                typename TBuffer,
                typename TWeight>
            __forceinline__ __device__
            void RelaxCTA_segment(TAppInst app_inst,
                          index_t seg_snode,
                          index_t seg_enode,
                          index_t seg_sedge_csr,
                          WorkSource work_source,
                          WorkTarget work_target_low,
                          WorkTarget work_target_high,
                          TBuffer current_priority,
                          PMAGraph csr_graph,   
                          GraphDatum<TValue> node_value_datum,
                          GraphDatum<TBuffer> node_buffer_datum,
                          GraphDatum<TWeight> edge_weight_datum)
            {
                const uint32_t tid = TID_1D;
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctor<TAppInst, PMAGraph, GraphDatum, TBuffer, TWeight>
                        push_functor(app_inst, work_target_low, work_target_high, current_priority,
                                     csr_graph, node_buffer_datum, edge_weight_datum);

                for (int i = 0 + tid; i < work_size_rup; i += nthreads)
                {
                    groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};

                    if (i < work_size)
                    {
                        const index_t node = work_source.get_work(i);
                        auto pair = app_inst.CombineValueBuffer(node,
                                                                node_value_datum.get_item_ptr(node),
                                                                node_buffer_datum.get_item_ptr(node));
                        if (pair.second)
                        {
                            np_local.start = csr_graph.begin_edge(node) - seg_sedge_csr;
                            np_local.size = csr_graph.end_edge(node) - seg_sedge_csr - np_local.start; // out-degree
                            Payload<TBuffer> payload;
                            payload.m_src = node;
                            payload.m_buffer_to_push = pair.first;
                            np_local.meta_data = payload;
                        }
                    }

                    switch (LB)
                    {
                        case LoadBalancing::COARSE_GRAINED:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                            schedule(np_local, push_functor);
                            break;
                        case LoadBalancing::FINE_GRAINED:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                            schedule(np_local, push_functor);
                            break;
                        case LoadBalancing::HYBRID:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                            schedule(np_local, push_functor);
                            break;
                        default:
                            assert(false);
                    }   
                }
            }

           template<LoadBalancing LB,
                    bool enable_priority,
                    typename TAppInst,
                    typename WorkSource,
                    typename PMAGraph,
                    typename TVec,
                    typename TValue,
                    typename TBuffer>
            __forceinline__ __device__
            void RelaxCTADBAdd(TAppInst app_inst,
                            index_t seg_snode,
                            index_t seg_enode,
                            uint64_t seg_sedge_csr,
                            bool zcflag,
                            WorkSource work_source,
                            PMAGraph vcsr_graph,
                            TValue *node_parent_datum,
                            TVec cache_edges_l2,
                            uint64_t *cache_size,
                            TValue *node_value_datum,
                            TBuffer * node_buffer_datum,
                            TBuffer current_priority,
                            BitmapDeviceObject out_active,
                            BitmapDeviceObject in_active)
            {
                const uint32_t tid = TID_1D;
                // if(tid==0)printf("RelaxCTADBAdd\n");
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctorAdd<TAppInst, PMAGraph, TValue, TBuffer,TVec>
                        push_functor(app_inst, vcsr_graph,  node_parent_datum, node_buffer_datum, cache_edges_l2,cache_size,out_active);
                for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
                {
                    groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};
                    if (tid < work_size)
                    {
                        const index_t node = work_source.get_work(tid);
                        // // printf("node %d tid %d#\n",node,tid);
                        bool L1 = (vcsr_graph.vertices_[node].virtual_start + vcsr_graph.vertices_[node].virtual_degree)  < (*cache_size) ? true : false;
                        bool L2 = vcsr_graph.vertices_[node].secondary_degree == (vcsr_graph.sync_vertices_[node].degree-vcsr_graph.vertices_[node].virtual_degree) ? true : false;
                        if((vcsr_graph.vertices_[node].virtual_degree > 0)&&L1&&L2){
                            np_local.start = vcsr_graph.begin_edge(node) + (uint64_t)vcsr_graph.vertices_[node].virtual_degree;
                            np_local.size = vcsr_graph.sync_vertices_[node].degree - vcsr_graph.vertices_[node].virtual_degree;
                            Payload<index_t> payload;
                            payload.m_src = node;
                            payload.m_buffer_to_push = node_buffer_datum[node];
                            // LOG("river_low l2 %d\n",*(vcsr_graph.river_low));
                            index_t node_index = atomicAdd(vcsr_graph.river_low, np_local.size);
                            vcsr_graph.vertices_[node].secondary_start = (uint64_t)node_index;
                            np_local.meta_data = payload;
                            // if(node == 5)
                            //     for(auto j = np_local.start; j < np_local.start + np_local.size; j++){
                            //         LOG("GPU %d %d\n",node,vcsr_graph.edges_[j]);
                            //     }
                        }
                        //如果增量更新的顶点被包含在缓存l1中，那么就刷新一下L1？
                        
                    }
                    switch (LB)
                    {
                        case LoadBalancing::COARSE_GRAINED:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                            schedule(np_local, push_functor,zcflag);
                            break;
                        case LoadBalancing::FINE_GRAINED:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                            schedule(np_local, push_functor,zcflag);
                            break;
                        case LoadBalancing::HYBRID:
                            groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                            schedule(np_local, push_functor,zcflag);
                            break;
                        default:
                            assert(false);
                    }
                }
            }



        //    template<LoadBalancing LB,
        //             bool enable_priority,
        //             typename TAppInst,
        //             typename WorkSource,
        //             typename PMAGraph,
        //             typename BufferVec,
        //             typename NodesHash,
        //             // template<typename> class GraphDatum,
        //             typename TValue,
        //             typename TBuffer,
        //             typename TWeight>
        //     __forceinline__ __device__
        //     void RelaxCTADBDelta(TAppInst app_inst,
        //                     index_t seg_snode,
        //                     index_t seg_enode,
        //                     uint64_t seg_sedge_csr,
        //                     bool zcflag,
        //                     WorkSource work_source,
        //                     PMAGraph vcsr_graph,
        //                     // bool* delta_array,
        //                     BufferVec buffer,
        //                     BufferVec buffer_l2,
        //                     NodesHash nodes_hash,
        //                     TValue *node_parent_datum,
        //                     TValue *node_value_datum,
        //                     TBuffer * node_buffer_datum,
        //                     // TValue *node_level_datum,
        //                     // GraphDatum<TWeight> edge_weight_datum,
        //                     TBuffer current_priority,
        //                     BitmapDeviceObject out_active,
        //                     BitmapDeviceObject in_active)
        //     {
        //         const uint32_t tid = TID_1D;
        //         // if(tid==0)printf("RelaxCTADBDelta\n");
        //         const uint32_t nthreads = TOTAL_THREADS_1D;
        //         const uint32_t work_size = work_source.get_size();
        //         const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
        //         PushFunctorDB<TAppInst, PMAGraph, BufferVec, NodesHash, TValue, TBuffer>
        //                 push_functor(app_inst, vcsr_graph, buffer, nodes_hash, node_parent_datum, node_buffer_datum, out_active);
        //         PushFunctorDBCachel2<TAppInst, PMAGraph, BufferVec, NodesHash, TValue, TBuffer>
        //                 push_functor_l1(app_inst, vcsr_graph, buffer, nodes_hash, node_parent_datum, node_buffer_datum, out_active);
        //         PushFunctorDBCachel2<TAppInst, PMAGraph, BufferVec, NodesHash, TValue, TBuffer>
        //                 push_functor_l2(app_inst, vcsr_graph, buffer_l2, nodes_hash, node_parent_datum, node_buffer_datum, out_active);
        //                 for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
        //                 {
        //                     groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};
        //                     groute::dev::np_local<Payload<TBuffer>> np_local_ca = {0, 0};
        //                     groute::dev::np_local<Payload<TBuffer>> np_local_secondary = {0, 0};
        //                     if (tid < work_size)
        //                     {
        //                         const index_t node = work_source.get_work(tid);
        //                         const auto pair = app_inst.CombineValueBuffer(node,
        //                         &node_value_datum[node],&node_buffer_datum[node]);
        //                         //s
        //                         // Value changed means validate combine, we need push the buffer to neighbors
        //                         if (pair.second)
        //                         {                                       
        //                             Payload<TBuffer> payload;
        //                             payload.m_src = node;
        //                             payload.m_buffer_to_push = pair.first;
        //                             // if(nodes_hash.contains(node)){
        //                             if(out_active.get_bit(node)){
        //                                 // LOG("static 缓存命中 %d\n",node);
        //                                 np_local_ca.start = vcsr_graph.vertices_[node].virtual_start;
        //                                 np_local_ca.size = vcsr_graph.vertices_[node].virtual_degree;
                                        
        //                                 np_local_secondary.start = vcsr_graph.vertices_[node].secondary_start;
        //                                 np_local_secondary.size = vcsr_graph.vertices_[node].secondary_degree;
        //                                 np_local_ca.meta_data  = payload;
        //                                 np_local_secondary.meta_data  = payload;
        //                             }else{
        //                                 LOG("static 缓存miss %d\n",node);
        //                                 // printf("--%d np_local.start %llu contains %d\n",node,np_local.start,nodes_hash.contains(node));
        //                                 np_local.start = vcsr_graph.begin_edge(node);
        //                                 np_local.size = vcsr_graph.vertices_[node].degree;
        //                                 // vcsr_graph.vertices_[node].virtual_start = atomicAdd(reinterpret_cast<unsigned long long *>(vcsr_graph.river), np_local.size);
        //                                 vcsr_graph.vertices_[node].virtual_start = vcsr_graph.atomicAdd(vcsr_graph.river, np_local.size);
        //                                 np_local.meta_data = payload;
        //                             }
        //                         }

        //                     } 
        //                     switch (LB)
        //                     {
                                
        //                         case LoadBalancing::COARSE_GRAINED:

        //                                 groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
        //                                 schedule(np_local_ca, push_functor_l1,zcflag);
        //                                 groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
        //                                 schedule(np_local_secondary, push_functor_l2,zcflag);
        //                                 groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
        //                                 schedule(np_local, push_functor,zcflag);
        //                             break;
        //                         case LoadBalancing::FINE_GRAINED:
        //                                 groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
        //                                 schedule(np_local_ca, push_functor_l1,zcflag);
        //                                 groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
        //                                 schedule(np_local_secondary, push_functor_l2,zcflag);
        //                                 groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
        //                                 schedule(np_local, push_functor,zcflag);
        //                             break;
        //                         case LoadBalancing::HYBRID:
        //                                 groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
        //                                 schedule(np_local_ca, push_functor_l1,zcflag);
        //                                 groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
        //                                 schedule(np_local_secondary, push_functor_l2,zcflag);
        //                                 groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
        //                                 schedule(np_local, push_functor,zcflag);
        //                             break;
        //                         default:
        //                             assert(false);
        //                     }

        //                 }
        //     }

           template<LoadBalancing LB,
                    bool enable_priority,
                    typename TAppInst,
                    typename WorkSource,
                    typename PMAGraph,
                    typename BufferVec,
                    typename TValue,
                    typename TBuffer>
            __forceinline__ __device__
            void RelaxCTADB_del(TAppInst app_inst,
                            index_t seg_snode,
                            index_t seg_enode,
                            uint64_t seg_sedge_csr,
                            bool zcflag,
                            WorkSource work_source,
                            PMAGraph vcsr_graph,
                            BufferVec buffer,
                            BufferVec buffer_l2,
                            uint64_t *count_gpu,
                            uint64_t *total_act_d,
                            bool *reset_node,
                            TValue *node_parent_datum,
                            TValue *node_value_datum,
                            TBuffer * node_buffer_datum,
                            TBuffer current_priority,
                            BitmapDeviceObject out_active,
                            BitmapDeviceObject in_active)
            {
                const uint32_t tid = TID_1D;
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctorDB_Del<TAppInst, PMAGraph, BufferVec, TValue, TBuffer>
                        push_functor(app_inst, vcsr_graph, buffer, node_parent_datum, node_buffer_datum, node_value_datum,out_active);
                PushFunctorDBCachel2_DEL<TAppInst, PMAGraph, BufferVec, TValue, TBuffer>
                        push_functor_l1(app_inst, vcsr_graph, buffer, node_parent_datum, node_buffer_datum,node_value_datum, out_active);
                for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
                {
                    groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};
                    groute::dev::np_local<Payload<TBuffer>> np_local_ca = {0, 0};
                    if (tid < work_size)
                    {
                        const index_t node = work_source.get_work(tid); 
                        vcsr_graph.vertices_[node].deletion = false;
                        reset_node[node]=true;
                        Payload<TBuffer> payload;
                        payload.m_src = node;
                        payload.m_buffer_to_push = UINT32_MAX;
                        if(vcsr_graph.vertices_[node].cache){
                            np_local_ca.start = vcsr_graph.vertices_[node].virtual_start;
                            np_local_ca.size = vcsr_graph.vertices_[node].virtual_degree;
                            np_local_ca.meta_data  = payload;
                            vcsr_graph.vertices_[node].hotness[0]+=1;
                            // total_act_d[node]++;
                            // total_act_d[node]+=np_local_ca.size;
                            
                        }else{
                            // total_act_d[node]++;
                            
                            // count_gpu[node]++;
                            np_local.start = vcsr_graph.begin_edge(node);
                            np_local.size = vcsr_graph.sync_vertices_[node].degree;
                            // total_act_d[node]+=np_local.size;
                            vcsr_graph.vertices_[node].hotness[0]+=1;
                            np_local.meta_data = payload;
                        }

                                
                    }  
                
                    switch (LB)
                    {
                        case LoadBalancing::COARSE_GRAINED:

                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                schedule(np_local_ca, push_functor_l1,zcflag);
                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                schedule(np_local, push_functor,zcflag);
                            break;
                        case LoadBalancing::FINE_GRAINED:
                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                                schedule(np_local_ca, push_functor_l1,zcflag);
                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                                schedule(np_local, push_functor,zcflag);
                            break;
                        case LoadBalancing::HYBRID:
                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                                schedule(np_local_ca, push_functor_l1,zcflag);
                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                                schedule(np_local, push_functor,zcflag);
                            break;
                        default:
                            assert(false);
                    }
                }
            }

           template<LoadBalancing LB,
                    bool enable_priority,
                    typename TAppInst,
                    typename WorkSource,
                    typename PMAGraph,
                    typename Hot,
                    typename BufferVec,
                    typename TValue,
                    typename TBuffer>
            __forceinline__ __device__
            void RelaxCTADB_all_vertices(TAppInst app_inst,
                            index_t seg_snode,
                            index_t seg_enode,
                            uint64_t seg_sedge_csr,
                            bool zcflag,
                            WorkSource work_source,
                            PMAGraph vcsr_graph,
                            Hot hotness,
                            BufferVec buffer,
                            BufferVec buffer_l2,
                            uint64_t *count_gpu,
                            uint64_t *total_act_d,
                            uint64_t* cache_size,
                            TValue* node_value_datum,
                            TValue* node_parent_datum,
                            TBuffer* node_buffer_datum,
                            TBuffer current_priority,
                            BitmapDeviceObject out_active,
                            BitmapDeviceObject in_active)
            {
                const uint32_t tid = TID_1D;
                const uint32_t nthreads = TOTAL_THREADS_1D;
                const uint32_t work_size = work_source.get_size();
                const uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
                PushFunctorDBADD<TAppInst, PMAGraph,BufferVec ,TValue,TBuffer>
                        push_functor(app_inst, vcsr_graph, buffer, node_parent_datum, node_buffer_datum,out_active);
                PushFunctorDBCachel2_ADD<TAppInst, PMAGraph,BufferVec ,TValue,TBuffer>
                        push_functor_l1(app_inst, vcsr_graph, buffer, node_parent_datum, node_buffer_datum,out_active);
                for (uint32_t i = 0 + tid; i < work_size_rup; i += nthreads)
                {
                    groute::dev::np_local<Payload<TBuffer>> np_local = {0, 0};
                    groute::dev::np_local<Payload<TBuffer>> np_local_ca = {0, 0};
                    if (tid < work_size)
                    {
                        const index_t node = work_source.get_work(tid);
                        TBuffer now_buff = atomicAdd(&node_buffer_datum[node],0);
                        if (now_buff!=UINT32_MAX)
                        {                                       
                            Payload<TBuffer> payload;
                            payload.m_src = node;
                            payload.m_buffer_to_push = now_buff;
                            // hotness.d_buffers[hotness.selector][node] +=1;
                            if(vcsr_graph.vertices_[node].cache){
                                np_local_ca.start = vcsr_graph.vertices_[node].virtual_start;
                                np_local_ca.size = vcsr_graph.vertices_[node].virtual_degree;
                                np_local_ca.meta_data  = payload;
                                // total_act_d[node]++;
                                // total_act_d[node]+=np_local_ca.size;
                                vcsr_graph.vertices_[node].hotness[0]++;
                            }else{
                                np_local.start = vcsr_graph.begin_edge(node);
                                np_local.size = vcsr_graph.sync_vertices_[node].degree;
                                np_local.meta_data = payload;
                                vcsr_graph.vertices_[node].hotness[0]++;
                                // total_act_d[node]++;
                                // total_act_d[node]+=np_local.size;
                                // count_gpu[node]++;
                            }

                            // printf("hotness %d\n",hotness.d_buffers[hotness.selector][node]);
                        }
                    } 
                    switch (LB)
                    {
                        case LoadBalancing::COARSE_GRAINED:
                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                schedule(np_local_ca, push_functor_l1,zcflag);
                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                schedule(np_local, push_functor,zcflag);
                            break;
                        case LoadBalancing::FINE_GRAINED:
                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                schedule(np_local_ca, push_functor_l1,zcflag);
                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_FINE_GRAINED>::template
                                schedule(np_local, push_functor,zcflag);
                            break;
                        case LoadBalancing::HYBRID:
                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_COARSE_GRAINED>::template
                                schedule(np_local_ca, push_functor_l1,zcflag);
                                groute::dev::CTAWorkSchedulerNew<Payload<TBuffer>, groute::dev::LB_HYBRID>::template
                                schedule(np_local, push_functor,zcflag);
                            break;
                        default:
                            assert(false);
                    }

                }
            }


        }
    }
}
#endif //HYBRID_ASYNC_PUSH_DD_H
