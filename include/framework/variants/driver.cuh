// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_DRIVER_H
#define HYBRID_DRIVER_H

#include <cub/grid/grid_barrier.cuh>
#include <groute/device/work_source.cuh>
#include <groute/graphs/csr_graph.cuh>
#include <utils/cuda_utils.h>
#include <groute/device/bitmap_impls.h>
#include <framework/variants/api.cuh>
#include <framework/common.h>
#include <framework/variants/async_push_td.cuh>
#include <framework/variants/async_push_dd.cuh>
#include <framework/variants/sync_push_td.cuh>
#include <framework/variants/sync_push_dd.cuh>
#include <framework/variants/async_pull_dd.cuh>
#include <framework/variants/async_pull_td.cuh>
#include <framework/variants/sync_pull_td.cuh>
#include <framework/variants/sync_pull_dd.cuh>
#include <framework/clion_cuda.cuh>
#define MAXL 4294967295
namespace sepgraph {
    namespace kernel {
        using sepgraph::common::LoadBalancing;

        template<typename WorkSource,
                typename TValue,
                typename TBuffer>
        __global__
        void PrintTable(WorkSource work_source,
                        groute::graphs::dev::GraphDatum<TValue> node_value_datum,
                        groute::graphs::dev::GraphDatum<TBuffer> node_buffer_datum) {
            const uint32_t tid = TID_1D;
            const uint32_t nthreads = TOTAL_THREADS_1D;
            const uint32_t work_size = work_source.get_size();

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                printf(" PrintTable %u %f %f\n", node, node_value_datum[i], node_buffer_datum[i]);
            }
        }
        template<typename TAppInst,
                typename WorkSource,
                typename PMAGraph>
        // template<typename TAppInst,
        //         typename WorkSource>
        __global__
        void RebuildWorklist_delta(TAppInst app_inst,
                                 WorkSource work_source,
                                PMAGraph vcsr_graph,
                                groute::dev::Queue<index_t> work_target) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                if (vcsr_graph.vertices_[node].delta) {
                    work_target.append(node);
                }
            }
        }

        template<typename WorkSource>
        __global__
        void PrintDegree(WorkSource work_source,
                         uint32_t *p_in_degree,
                         uint32_t *p_out_degree) {
            uint32_t tid = TID_1D;
            uint32_t work_size = work_source.get_size();

            if (tid == 0) {
                for (int i = 0; i < work_size; i++) {
                    index_t node = work_source.get_work(i);

                    printf("node: %u in-degree: %u out-degree: %u\n",
                           node,
                           p_in_degree[node],
                           p_out_degree[node]);
                }
            }
        }

        template<typename PMA,typename Cache,typename NodeHash>
        __global__
        void PrintCacheGraph(PMA vcsr_graph,
                         Cache vec,
                         NodeHash nodes_hash) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            for (uint32_t i = 0 + tid; i < vcsr_graph.nnodes; i += nthreads) {
                if(nodes_hash.find(i)){
                    auto start = vcsr_graph.vertices_[i].virtual_start;
                    auto size = vcsr_graph.vertices_[i].virtual_degree;
                        for(auto j = start; j < start + size;j++){
                            printf("%lu %lu\n",i,vec[j]);
                        }
                }
            }
        }
        template<typename PMA,typename Cache,typename NodeHash>
        __global__
        void PrintCacheEdges(PMA vcsr_graph,
                         Cache vec,
                         NodeHash nodes_hash) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            for (uint64_t i = 0 + tid; i < *vcsr_graph.river; i += nthreads) {
                    printf("%d edge %d\n",i,vec[i]);
            }
        }

        template<typename TAppInst,
                typename PMAGraph,
                typename WorkSource,
                typename TDB_8>
        __global__
        void comp_hotness(TAppInst app_inst,
                            PMAGraph vcsr_graph,
                              WorkSource work_source,TDB_8 d_hotness) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                d_hotness.d_buffers[d_hotness.selector][node] = 1/d_hotness.d_buffers[d_hotness.selector][node];
                
            }
        }
        template<typename TAppInst,
                typename PMAGraph,
                typename WorkSource>
        __global__
        void search_batch(TAppInst app_inst,
                            PMAGraph vcsr_graph,
                              WorkSource work_source,
                              uint64_t *cache_size,
                              index_t* d_sum,
                                uint32_t* d_id) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            uint64_t space = (*cache_size);
            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t j = work_source.get_work(i);
                // bool cache = vcsr_graph.vertices_[d_id[j]].cache;
                //不在缓存里 && 让他delta=true
                if(d_sum[j+1]<(space)){
                //     // printf("v_id %d d_sum[%d] = %d\n",d_id[j],j,d_sum[j]);
                    vcsr_graph.vertices_[d_id[j]].delta = true;
                    // vcsr_graph.vertices_[d_id[j]].virtual_start = d_sum[j];
                //     vcsr_graph.vertices_[d_id[j]].cache = false;
                }
                    // printf("d_id[%d] = %d\n",j,d_id[j]);
            }
        }
         template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename PMAGraph,
                typename BufferVec,
                typename TValue,
                typename TBuffer>
        __global__
        void SyncPushDDBAmend_cache(TAppInst app_inst,
                         index_t seg_snode,
			             index_t seg_enode,
			             uint64_t seg_sedge_csr,
			             bool zcflag,
                         WorkSource work_source,
                         const PMAGraph vcsr_graph,
                         BufferVec *buffer,
                         BufferVec *buffer_l2,
                         uint64_t *cache_size,
                        TValue *node_parent_datum,
                        TValue *node_value_datum,
                        TBuffer * node_buffer_datum,
                        // TBuffer * node_tmp_buffer_datum,
                         BitmapDeviceObject out_active,
                         BitmapDeviceObject in_active) {
                sync_push_dd::RelaxCTADBAmend_cache<LB, false>(app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                                                   work_source,
                                                   vcsr_graph,
                                                   buffer,
                                                   buffer_l2,
                                                   cache_size,
                                                   node_parent_datum,
                                                   node_value_datum,
                                                   node_buffer_datum,
                                                //    node_tmp_buffer_datum,
                                                   (TBuffer) 0,
                                                   out_active,
                                                   in_active);
        }
        // TODO ALL replace get_item with operator[]
        template<typename WorkSource,
                typename TValue,
                typename TBuffer,
                typename TWeight,
                typename TDB_32,
                 typename TDB_8,
                template<typename, typename, typename, typename ...> class TAppImpl,
                typename... UnusedData>
        __global__
        void InitGraph(TAppImpl<TValue, TBuffer, TWeight, UnusedData...> app_inst,
                       WorkSource work_source,
                       TDB_32  d_id,
                       TDB_8   d_hotness,
                       TValue *node_parent_datum,
                       TValue *node_value_datum,
                       TBuffer *node_buffer_datum) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                TBuffer init_buffer = app_inst.GetInitBuffer(node);
                node_parent_datum[node] = node;
                d_id.d_buffers[d_id.selector][node] = node;
                d_hotness.d_buffers[d_hotness.selector][node] = 0;
                node_value_datum[node] = app_inst.GetInitValue(node);
                node_buffer_datum[node] = init_buffer;
            }
        }

        template<typename TAppInst,
                typename PMAGraph,
                typename WorkSource,
                typename TDB_8>
        __global__
        void flush_cache(TAppInst app_inst,
                            PMAGraph vcsr_graph,
                              WorkSource work_source,TDB_8 d_hotness) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
               if(vcsr_graph.vertices_[node].cache){
                if(vcsr_graph.vertices_[node].delta){
                    vcsr_graph.vertices_[node].delta = false;
                }else{
                    vcsr_graph.vertices_[node].virtual_degree = 0;
                    vcsr_graph.vertices_[node].virtual_start = 0;
                    vcsr_graph.vertices_[node].cache= false;
                }
               }
                
            }
        }

        template<typename WorkSource>
        __global__
        void copy_cache(index_t* edge_l1,
                            index_t* edge_l3,
                              WorkSource work_source) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t edge = work_source.get_work(i);
                edge_l1[edge] = edge_l3[edge];
                edge_l3[edge] = 0;
            }
        }

        template<template<typename, typename, typename, typename ...> class TAppImpl,
                typename PMAGraph,
                typename TValue,
                typename TBuffer,
                typename TWeight,
                typename TWEdge,
                typename... UnusedData>
        __global__
        void reset_pr_del_edges(
                        TAppImpl<TValue, TBuffer, TWeight, UnusedData...> app_inst,
                        PMAGraph vcsr_graph,
                       TWEdge *del_edges_d,
                       uint32_t *work_size_d) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_size_d[1];
            uint32_t loca_begin = work_size_d[0];
            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) 
            {
                // if(i>=loca_begin&&i<work_size){
                    index_t node = del_edges_d[i+loca_begin].u;
                    // printf("active node %d#\n",node);
                    vcsr_graph.vertices_[node].cache = false;
                    vcsr_graph.vertices_[node].virtual_start = 0;
                    vcsr_graph.vertices_[node].virtual_degree = 0;
                    // vcsr_graph.vertices_[node].secondary_degree = 0;       
                // }

            }

        }
        template<template<typename, typename, typename, typename ...> class TAppImpl,
                typename PMAGraph,
                typename TValue,
                typename TBuffer,
                typename TWeight,
                typename TWEdge,
                typename... UnusedData>
        __global__
        void reset_pr_add_edges(
                        TAppImpl<TValue, TBuffer, TWeight, UnusedData...> app_inst,
                        PMAGraph vcsr_graph,
                       TWEdge *add_edges_d,
                       uint32_t *work_size_d) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_size_d[1];
            uint32_t loca_begin = work_size_d[0];
            uint32_t work_size_rup = round_up(work_size, blockDim.x) * blockDim.x;
            for (uint32_t i = loca_begin + tid; i < loca_begin+work_size_rup; i += nthreads) 
            {
                if(tid < work_size){
                    index_t node = add_edges_d[tid].u;
                    printf("add node %d#\n",node);
                    vcsr_graph.vertices_[node].delta = true;
                    vcsr_graph.vertices_[node].virtual_degree = 0;
                    vcsr_graph.vertices_[node].secondary_degree = 0;
                }

            }

        }

         template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename PMAGraph,
                typename BufferVec,
                typename TValue,
                typename TBuffer>
        __global__
        void RunSyncPushDDB_MFC(TAppInst app_inst,
                         index_t seg_snode,
			             index_t seg_enode,
			             uint64_t seg_sedge_csr,
			             bool zcflag,
                         WorkSource work_source,
                         const PMAGraph vcsr_graph,
                         BufferVec *buffer,
                         BufferVec *buffer_l2,
                        uint64_t *count_gpu,
                         uint64_t *transfer,
                         uint64_t *cache_size,
                        TValue *node_parent_datum,
                        TValue *node_value_datum,
                        TBuffer * node_buffer_datum,
                        BitmapDeviceObject out_active,
                        BitmapDeviceObject in_active) {
                sync_push_dd::RelaxCTADBMFC<LB, false>(app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                                                   work_source,
                                                   vcsr_graph,
                                                   buffer,
                                                   buffer_l2,
                                                   count_gpu,
                                                   transfer,
                                                   cache_size,
                                                   node_parent_datum,
                                                   node_value_datum,
                                                   node_buffer_datum,
                                                   (TBuffer) 0,
                                                   out_active,
                                                   in_active);
        }      
        template<typename WorkSource>
        __global__
        void reset_cache(
                       WorkSource work_source,
                       uint64_t *count_gpu) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                count_gpu[node]=0;
            }
        }
        template<template<typename, typename, typename, typename ...> class TAppImpl,
                typename PMAGraph,
                typename TValue,
                typename TBuffer,
                typename TWEdge,
                typename TWeight,
                typename BufferVec,
                // typename NodesHash,
                typename... UnusedData>
        __global__
        void reset_del_edges(
                        TAppImpl<TValue, TBuffer, TWeight, UnusedData...> app_inst,
                        PMAGraph vcsr_graph,
                        TValue *node_parent_datum,
                       TValue *node_value_datum,
                       TBuffer *node_buffer_datum,
                       BufferVec buffer,
                       BufferVec buffer_l2,
                       TWEdge *del_edges_d,
                       uint32_t *work_size_d,
                       bool* reset_nodes) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_size_d[1];
            uint32_t loca_begin = work_size_d[0];
            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) 
            {
                index_t src = del_edges_d[i+loca_begin].u;
                index_t dst = del_edges_d[i+loca_begin].v;
                if(node_parent_datum[dst]==src){
                    TBuffer init_buffer = app_inst.GetInitBuffer(dst);
                    node_buffer_datum[dst] = init_buffer;
                    node_value_datum[dst] = app_inst.GetInitValue(dst);
                    node_parent_datum[dst] = UINT32_MAX;
                    vcsr_graph.vertices_[dst].deletion = true;
                    // reset_nodes[dst]=true;
                    
                }

            }

        }

        template<typename TAppInst,
                typename PMAGraph,
                typename WorkSource,
                typename TBuffer,
                typename TDB_8>
        __global__
        void comp_hotness_pr(TAppInst app_inst,
                            PMAGraph vcsr_graph,
                              WorkSource work_source,
                              TBuffer buffer_datum,
                              TDB_8 d_hotness) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                    d_hotness.d_buffers[d_hotness.selector][node] = ((vcsr_graph.vertices_[node].hotness[0]+vcsr_graph.vertices_[node].hotness[1]+vcsr_graph.vertices_[node].hotness[2]+vcsr_graph.vertices_[node].hotness[3]));
                    // d_hotness.d_buffers[d_hotness.selector][node] = (vcsr_graph.vertices_[node].hotness[0]);
                
            }
        }
        template<typename TAppInst,
                typename PMAGraph,
                typename WorkSource,
                typename TBuffer,
                typename TDB_8>
        __global__
        void comp_hotness_sssp(TAppInst app_inst,
                            PMAGraph vcsr_graph,
                              WorkSource work_source,
                              TBuffer buffer_datum,
                              TDB_8 d_hotness) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);

                if(buffer_datum[node]==UINT32_MAX){
                    d_hotness.d_buffers[d_hotness.selector][node] = 0;
                }else{
                    d_hotness.d_buffers[d_hotness.selector][node] = ((vcsr_graph.vertices_[node].hotness[0]+vcsr_graph.vertices_[node].hotness[1]+vcsr_graph.vertices_[node].hotness[2]+vcsr_graph.vertices_[node].hotness[3]));
                }
                // d_hotness.d_buffers[d_hotness.selector][node] =vcsr_graph.sync_vertices_[node].degree;
                    // d_hotness.d_buffers[d_hotness.selector][node] = (vcsr_graph.vertices_[node].hotness[0]);
                
            }
        }
        template<typename TAppInst,
                typename PMAGraph,
                typename WorkSource,
                typename TDB_32>
        __global__
        void extract_vtx_degree(TAppInst app_inst,
                            PMAGraph vcsr_graph,
                              WorkSource work_source,TDB_32 d_id,index_t *d_v) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                d_v[node] = vcsr_graph.sync_vertices_[d_id.d_buffers[d_id.selector][node]].degree;
                // if(node < 100)printf("d_d[%d] degree %d sync_v[%d]\n",node,vcsr_graph.sync_vertices_[d_id.d_buffers[d_id.selector][node]].degree,d_id.d_buffers[d_id.selector][node]);
            }
        }

        template<typename TAppInst,
                typename PMAGraph,
                typename WorkSource,
                typename TDB_8>
        __global__
        void reset_hotness(TAppInst app_inst,
                            PMAGraph vcsr_graph,
                              WorkSource work_source,TDB_8 d_hotness) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                vcsr_graph.vertices_[node].hotness[3] = vcsr_graph.vertices_[node].hotness[2];
                vcsr_graph.vertices_[node].hotness[2] = vcsr_graph.vertices_[node].hotness[1];
                vcsr_graph.vertices_[node].hotness[1] = vcsr_graph.vertices_[node].hotness[0];
                vcsr_graph.vertices_[node].hotness[0] = 0;
            }
        }

        /**
         * @brief Quaful reset_del_edges()
         * @tparam TValue  node_parent_datum
         * @tparam TValue  node_value_datum
         * @tparam TBuffer  node_buffer_datum
         * @tparam TWEdge  added_edges_d
         * @tparam uint32_t  work_size_d
         */
        template<template<typename, typename, typename, typename ...> class TAppImpl,
                typename PMAGraph,
                typename TValue,
                typename TBuffer,
                typename TWEdge,
                typename TWeight,
                typename... UnusedData>
        __global__
        void reset_add_edges(
                        TAppImpl<TValue, TBuffer, TWeight, UnusedData...> app_inst,
                        PMAGraph vcsr_graph,
                        TValue *node_parent_datum,
                       TValue *node_value_datum,
                       TBuffer *node_buffer_datum,
                       TWEdge *add_edges_d,
                       uint32_t *work_size_d) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_size_d[1];
            uint32_t loca_begin = work_size_d[0];
            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) 
            {
                index_t node = add_edges_d[i+loca_begin].u;
                vcsr_graph.vertices_[node].cache=false;
                vcsr_graph.vertices_[node].virtual_start = 0;
                vcsr_graph.vertices_[node].virtual_degree = 0;
            }
        }

        template<typename PMAGraph, typename WorkSource>
        __global__ void InitDegree(PMAGraph csr_graph,
                                   WorkSource work_source,
                                   uint32_t *p_in_degree,
                                   uint32_t *p_out_degree) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();

            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                index_t begin_edge = csr_graph.begin_edge(node),
                        end_edge = csr_graph.end_edge(node);

                p_out_degree[node] = end_edge - begin_edge;

                for (int edge = begin_edge; edge < end_edge; edge++) {
                    index_t dest = csr_graph.edge_dest(edge);

                    atomicAdd(&p_in_degree[dest], 1);
                }
            }
        }


        template<typename TQueue, typename TBitmap>
        __global__ void QueueToBitmap(TQueue queue, TBitmap bitmap) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = queue.count();

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
                bitmap.set_bit_atomic(queue.read(i));
            }
        }

        template<typename TBitmap, typename TQueue>
        __global__ void BitmapToQueue(TBitmap bitmap, TQueue queue) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = bitmap.get_size();

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
                if (bitmap.get_bit(i)) {
                    queue.append(i);
                }
            }
        }
        
        
        template<typename TBitmap, typename TQueue>
        __global__ void BitmapToQueueRange(TBitmap bitmap, TQueue queue,index_t start,index_t end) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = bitmap.get_size();

            for (uint32_t i = 0 + tid; i < end - start; i += nthreads) {
                
	               index_t pos = start + i;
                if (bitmap.get_bit(pos)) {
		    //printf("node: %d\n",i);
                    queue.append(pos);
                }
            }
        }

        template<typename WorkSource, typename TBuffer>
        __global__ void Sample(WorkSource work_source,
                               groute::graphs::dev::GraphDatum<TBuffer> node_buffer_datum,
                               TBuffer *p_sampled_values) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();

            for (uint32_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);

                p_sampled_values[i] = node_buffer_datum.get_item(node);
            }
        }

        template<typename TVec, typename PMAGraph>
        __global__ void reset_flush_edges(TVec *cache_l1,
                       TVec *cache_l3,PMAGraph vcsr_graph,uint64_t* cache_size) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            if(tid < (*cache_size)){
                cache_l1[tid] = cache_l3[tid];
                // printf("tid %d river %d\n",tid,vcsr_graph.river);
            } 
        }

        template<typename TAppInst,
                typename WorkSource,
                // template<typename> class GraphDatum,
                typename TBuffer,
                typename TValue>
        // template<typename TAppInst,
        //         typename WorkSource>
        __global__
        void RebuildWorklist(TAppInst app_inst,
                              WorkSource work_source,
                              groute::dev::Queue<index_t> work_target,
                            //   GraphDatum<TBuffer> node_buffer_datum,
                            //   GraphDatum<TValue> node_value_datum,
                            TBuffer *node_buffer_datum,
                            TValue *node_value_datum) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                // activeNodesLabeling[node] = 0;
                // activeNodesDegree[node] = 0;
                if (app_inst.IsActiveNode(node, node_buffer_datum[node], node_value_datum[node])) {
                    // printf("active node %d\n",node);
                    work_target.append(node);
                }
            }
        }

        template<typename TAppInst, typename PMAGraph,
                typename WorkSource>
        // template<typename TAppInst,
        //         typename WorkSource>
        __global__
        void copy_index(TAppInst app_inst,
                            PMAGraph vcsr_graph,
                              WorkSource work_source) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                vcsr_graph.vertices_[node].virtual_degree = vcsr_graph.vertices_[node].third_degree;
                vcsr_graph.vertices_[node].virtual_start = vcsr_graph.vertices_[node].third_start;
                vcsr_graph.vertices_[node].third_degree =0;
                vcsr_graph.vertices_[node].third_start = 0;

            }
        }

        template<typename TAppInst,
                typename WorkSource,
                // template<typename> class GraphDatum,
                typename TBuffer,
                typename TValue>
        // template<typename TAppInst,
        //         typename WorkSource>
        __global__
        void RebuildWorklistAllVertices(TAppInst app_inst,
                              WorkSource work_source,
                              groute::dev::Queue<index_t> work_target,
                            //   GraphDatum<TBuffer> node_buffer_datum,
                            //   GraphDatum<TValue> node_value_datum,
                            TBuffer *node_buffer_datum,
                            TValue *node_value_datum) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                // assert(node<105153952);
                work_target.append(node);
            }
        }

        template<typename TAppInst,
                typename WorkSource,
                typename TBuffer,
                typename TValue>
        __global__
        void RebuildWorklist_INC(TAppInst app_inst,
                              WorkSource work_source,
                              groute::dev::Queue<index_t> work_target,
                              TBuffer* node_buffer_datum,
                              TValue* node_value_datum){
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                if (app_inst.IsActiveNode(node, node_buffer_datum[node], node_value_datum[node]) || app_inst.IsActiveNode_INC(node, node_buffer_datum[node], node_value_datum[node])) {
                    work_target.append(node);
                }
            }
        }

        template<typename TAppInst,
                typename PMAGraph,
                typename WorkSource>
        __global__
        void RebuildWorklist_Identify(TAppInst app_inst,
                            PMAGraph vcsr_graph,
                              WorkSource work_source,
                              groute::dev::Queue<index_t> work_target) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                if (vcsr_graph.vertices_[node].cache) {
                        work_target.append(node);
                }
            }
        }

        template<typename TAppInst,
                typename PMAGraph,
                typename WorkSource>
        __global__
        void RebuildWorklist_evition(TAppInst app_inst,
                            PMAGraph vcsr_graph,
                              WorkSource work_source,
                              groute::dev::Queue<index_t> work_target) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                if (vcsr_graph.vertices_[node].cache) {
                    // work_target.append(node);
                    if(vcsr_graph.vertices_[node].delta){
                        vcsr_graph.vertices_[node].delta = false;
                    }else{
                        vcsr_graph.vertices_[node].virtual_start = 0;
                        vcsr_graph.vertices_[node].virtual_degree= 0;
                        vcsr_graph.vertices_[node].cache = false;
                    }
                }
            }
        }

        template<typename TAppInst,
                typename PMAGraph,
                typename WorkSource>
        __global__
        void RebuildWorklist_evition_v2(TAppInst app_inst,
                            PMAGraph vcsr_graph,
                              WorkSource work_source,
                              groute::dev::Queue<index_t> work_target) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                if (vcsr_graph.vertices_[node].cache) {
                    // work_target.append(node);
                    // if(vcsr_graph.vertices_[node].delta){
                    //     vcsr_graph.vertices_[node].delta = false;
                    // }else{
                        vcsr_graph.vertices_[node].virtual_start = 0;
                        vcsr_graph.vertices_[node].virtual_degree= 0;
                        vcsr_graph.vertices_[node].cache = false;
                        *vcsr_graph.river = 0;
                }
            }
        }

        template<typename TAppInst,
                typename WorkSource,
                typename PMAGraph,
                typename TBuffer,
                typename TValue>
        // template<typename TAppInst,
        //         typename WorkSource>
        __global__
        void RebuildWorklist_add(TAppInst app_inst,
                                 WorkSource work_source,
                                PMAGraph vcsr_graph,
                                groute::dev::Queue<index_t> work_target,
                                TBuffer *node_buffer_datum,
                                TValue *node_value_datum) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                if (vcsr_graph.vertices_[node].delta) {
                    work_target.append(node);
                }
            }
        }
        template<typename TAppInst,
                typename WorkSource,
                typename PMAGraph,
                typename TBuffer,
                typename TValue>
        // template<typename TAppInst,
        //         typename WorkSource>
        __global__
        void RebuildWorklist_deletion(TAppInst app_inst,
                                 WorkSource work_source,
                                PMAGraph vcsr_graph,
                                groute::dev::Queue<index_t> work_target,
                                TBuffer *node_buffer_datum,
                                TValue *node_value_datum) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                if (vcsr_graph.vertices_[node].deletion){
                    work_target.append(node);
                }
            }
        }
        template<typename TAppInst,
                typename PMAGraph,
                typename WorkSource,
                typename TDB_32,
                typename TBuffer,
                typename TValue>
        // template<typename TAppInst,
        //         typename WorkSource>
        __global__
        void RebuildWorklist_rd(TAppInst app_inst,
                                PMAGraph vcsr_graph,
                              WorkSource work_source,
                              groute::dev::Queue<index_t> work_target,
                            //   GraphDatum<TBuffer> node_buffer_datum,
                            //   GraphDatum<TValue> node_value_datum,
                            TDB_32 hotness,
                            TBuffer *node_buffer_datum,
                            TValue *node_value_datum) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                hotness.d_buffers[hotness.selector][node] = 0;
                vcsr_graph.vertices_[node].deletion = false;
            }
        }

        template<typename TAppInst,
        typename PMAGraph,
                typename WorkSource>
        // template<typename TAppInst,
        //         typename WorkSource>
        __global__
        void RebuildWorklist_del(TAppInst app_inst,
                            PMAGraph vcsr_graph,
                              WorkSource work_source,
                              groute::dev::Queue<index_t> work_target) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                if (vcsr_graph.vertices_[node].deletion) {
                    work_target.append(node);
                }
            }
        }

        template<typename TAppInst,
                typename WorkSource,
                // template<typename> class GraphDatum,
                typename TBuffer,
                typename TValue>
        // template<typename TAppInst,
        //         typename WorkSource>
        __global__
        void RebuildWorklist_compaction(TAppInst app_inst,
                              WorkSource work_source,
                              groute::dev::Queue<index_t> work_target,
                              TBuffer *node_buffer_datum,
                              TValue *node_value_datum,
                              uint32_t *activeNodesLabeling,
                              uint32_t *activeNodesDegree,
                              uint32_t *p_out_degree) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();

            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);
                activeNodesLabeling[node] = 0;
                activeNodesDegree[node] = 0;
                if (app_inst.IsActiveNode(node, node_buffer_datum[node], node_value_datum[node])) {
                    work_target.append(node);
                    activeNodesLabeling[node] = 1;
                    activeNodesDegree[node] = p_out_degree[node];
                }
            }
        }

        __global__ void makeQueue(uint32_t *activeNodes, uint32_t *activeNodesLabeling,
                                    uint32_t *prefixLabeling, uint32_t numNodes)
        {
            uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
            if(id < numNodes && activeNodesLabeling[id] == 1){
                activeNodes[prefixLabeling[id]] = id;
            }
        }

        __global__ void makeActiveNodesPointer(uint32_t *activeNodesPointer, uint32_t *activeNodesLabeling, 
                                                    uint32_t *prefixLabeling, uint32_t *prefixSumDegrees, 
                                                    uint32_t numNodes)
        {
            uint32_t id = blockDim.x * blockIdx.x + threadIdx.x;
            if(id < numNodes && activeNodesLabeling[id] == 1){
                activeNodesPointer[prefixLabeling[id]] = prefixSumDegrees[id];
            }
        }



        template<typename TAppInst,
                 typename WorkSource,
		        //  template<typename> class GraphDatum,
                 typename TValue,
                 typename TBuffer>
        __global__ void SumResQueue(TAppInst app_inst,
                                     WorkSource work_source,
                                     TValue *node_value_datum,
                                     TBuffer *node_buffer_datum,
                                     TValue *p_total_res) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            TValue local_sum = 0.0;
            typedef cub::WarpReduce<TValue> WarpReduce;
            __shared__ typename WarpReduce::TempStorage temp_storage[8];

            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);

                local_sum += app_inst.sum_value(node, node_value_datum[node], node_buffer_datum[node]);

            }
	    
            int warp_id = threadIdx.x / 32;
            TValue aggregate = WarpReduce(temp_storage[warp_id]).Sum(local_sum);

            if (cub::LaneId() == 0) {
	       
                atomicAdd(p_total_res, aggregate);
		
            }
        }
          
        
        template<typename WorkSource>
        __global__ void SumOutDegreeQueue(WorkSource work_source,
                                     uint32_t *p_out_degree,
                                     uint32_t *p_total_out_degree) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            uint32_t local_sum = 0;
            typedef cub::WarpReduce<uint32_t> WarpReduce;
            __shared__ typename WarpReduce::TempStorage temp_storage[8];

            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = work_source.get_work(i);

                local_sum += p_out_degree[node];
		
            }
	    
            int warp_id = threadIdx.x / 32;
            int aggregate = WarpReduce(temp_storage[warp_id]).Sum(local_sum);

            if (cub::LaneId() == 0) {
                atomicAdd(p_total_out_degree, aggregate);
		
            }
        }

        template <typename TBitmap>
        __global__ void SumOutDegreeBitmap(TBitmap work_source,
                                     uint32_t *p_out_degree,
                                     uint32_t *p_total_out_degree) {
            uint32_t tid = TID_1D;
            uint32_t nthreads = TOTAL_THREADS_1D;
            uint32_t work_size = work_source.get_size();
            uint32_t local_sum = 0;
            typedef cub::WarpReduce<uint32_t> WarpReduce;
            __shared__ typename WarpReduce::TempStorage temp_storage[8];


            for (index_t i = 0 + tid; i < work_size; i += nthreads) {
                index_t node = i;

                if (work_source.get_bit(node)) {
                    local_sum += p_out_degree[node];
                }
            }

            int warp_id = threadIdx.x / 32;
            int aggregate = WarpReduce(temp_storage[warp_id]).Sum(local_sum);

            if (cub::LaneId() == 0) {
                atomicAdd(p_total_out_degree, local_sum);
            }
        }

        template<typename TAppInst>
        __global__
        void CallPostComputation(TAppInst app_inst) {
            uint32_t tid = TID_1D;

            if (tid == 0) {
                app_inst.PostComputation();
            }
        }

        template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename PMAGraph,
                template<typename> class GraphDatum,
                typename TValue,
                typename TBuffer,
                typename TWeight>
        __global__
        void AsyncPushTD(TAppInst app_inst,
                         WorkSource work_source,
                         PMAGraph csr_graph,
                         GraphDatum<TValue> node_value_datum,
                         GraphDatum<TBuffer> node_buffer_datum,
                         GraphDatum<TWeight> edge_weight_datum) {
            if (LB == LoadBalancing::NONE) {
                async_push_td::Relax<false>(app_inst,
                                            work_source,
                                            csr_graph,
                                            node_value_datum,
                                            node_buffer_datum,
                                            edge_weight_datum,
                                            (TBuffer) 0);
            } else {
                async_push_td::RelaxCTA<LB, false>(app_inst,
                                                   work_source,
                                                   csr_graph,
                                                   node_value_datum,
                                                   node_buffer_datum,
                                                   edge_weight_datum,
                                                   (TBuffer) 0);
            }
        }
        template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename PMAGraph,
                template<typename> class GraphDatum,
                typename TValue,
                typename TBuffer,
                typename TWeight>
        __global__
        void SyncPushTD(TAppInst app_inst,
                         WorkSource work_source,
                         PMAGraph csr_graph,
                         GraphDatum<TValue> node_value_datum,
                         GraphDatum<TBuffer> node_buffer_datum,
                         GraphDatum<TWeight> edge_weight_datum) {
            if (LB == LoadBalancing::NONE) {
                sync_push_td::Relax<false>(app_inst,
                                            work_source,
                                            csr_graph,
                                            node_value_datum,
                                            node_buffer_datum,
                                            edge_weight_datum,
                                            (TBuffer) 0);
            } else {
                sync_push_td::RelaxCTA<LB, false>(app_inst,
                                                   work_source,
                                                   csr_graph,
                                                   node_value_datum,
                                                   node_buffer_datum,
                                                   edge_weight_datum,
                                                   (TBuffer) 0);
            }
        }

        template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename PMAGraph,
                template<typename> class GraphDatum,
                typename TValue,
                typename TBuffer,
                typename TWeight>
        __global__
        void AsyncPushTDPrio(TAppInst app_inst,
                             WorkSource work_source,
                             PMAGraph csr_graph,
                             GraphDatum<TValue> node_value_datum,
                             GraphDatum<TBuffer> node_buffer_datum,
                             GraphDatum<TWeight> edge_weight_datum,
                             TBuffer current_priority) {
            if (LB == LoadBalancing::NONE) {
                async_push_td::Relax<true>(app_inst,
                                           work_source,
                                           csr_graph,
                                           node_value_datum,
                                           node_buffer_datum,
                                           edge_weight_datum,
                                           current_priority);
            } else {
                async_push_td::RelaxCTA<LB, true>(app_inst,
                                                  work_source,
                                                  csr_graph,
                                                  node_value_datum,
                                                  node_buffer_datum,
                                                  edge_weight_datum,
                                                  current_priority);
            }
        }

        template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename PMAGraph,
                template<typename> class GraphDatum,
                typename TValue,
                typename TBuffer,
                typename TWeight>
        __global__
        void AsyncPushTDFused(TAppInst app_inst,
                              WorkSource work_source,
                              PMAGraph csr_graph,
                              GraphDatum<TValue> node_value_datum,
                              GraphDatum<TBuffer> node_buffer_datum,
                              GraphDatum<TWeight> edge_weight_datum,
                              cub::GridBarrier grid_barrier,
                              uint32_t *p_active_count) {
            uint32_t work_size = work_source.get_size();
            uint32_t tid = TID_1D;
                         if(tid==0)printf("AsyncPushTDFused\n");           
            while (*p_active_count) {
                if (LB == LoadBalancing::NONE) {
                    async_push_td::Relax<false>(app_inst,
                                                work_source,
                                                csr_graph,
                                                node_value_datum,
                                                node_buffer_datum,
                                                edge_weight_datum,
                                                (TBuffer) 0);
                } else {
                    async_push_td::RelaxCTA<LB, false>(app_inst,
                                                       work_source,
                                                       csr_graph,
                                                       node_value_datum,
                                                       node_buffer_datum,
                                                       edge_weight_datum,
                                                       (TBuffer) 0);
                }

                grid_barrier.Sync();

                if (tid == 0) {
//                        printf("Round: %u Policy to execute: ASYNC_PUSH_TD In: %u Out: %u",
//                               *app_inst.m_p_current_round, work_size, *p_active_count);
                    app_inst.PostComputation();
                    *p_active_count = 0;
                    *app_inst.m_p_current_round += 1;
                }
                grid_barrier.Sync();

                common::CountActiveNodes(app_inst,
                                         work_source,
                                         node_buffer_datum,
                                         p_active_count);
                grid_barrier.Sync();
            }

            // fix the last iteration times
            if (tid == 0) {
                *app_inst.m_p_current_round -= 1;
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
        __global__
        void AsyncPushDD(TAppInst app_inst,
                         WorkSource work_source,
                         WorkTarget work_target,
                         PMAGraph csr_graph,
                         GraphDatum<TValue> node_value_datum,
                         GraphDatum<TBuffer> node_buffer_datum,
                         GraphDatum<TWeight> edge_weight_datum) {
                            uint32_t tid = TID_1D;
                            if(tid==0)printf("AsyncPushDD\n");    
            if (LB == LoadBalancing::NONE) {
                async_push_dd::Relax(app_inst,
                                     work_source,
                                     work_target,
                                     work_target,
                                     (TBuffer) 0,
                                     csr_graph,
                                     node_value_datum,
                                     node_buffer_datum,
                                     edge_weight_datum);
            } else {
                async_push_dd::RelaxCTA<LB>(app_inst,
                                            work_source,
                                            work_target,
                                            work_target,
                                            (TBuffer) 0,
                                            csr_graph,
                                            node_value_datum,
                                            node_buffer_datum,
                                            edge_weight_datum);
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
        __global__
        void AsyncPushDDPrio(TAppInst app_inst,
                             WorkSource work_source,
                             WorkTarget work_target_low,
                             WorkTarget work_target_high,
                             TBuffer current_priority,
                             PMAGraph csr_graph,
                             GraphDatum<TValue> node_value_datum,
                             GraphDatum<TBuffer> node_buffer_datum,
                             GraphDatum<TWeight> edge_weight_datum) {
                                uint32_t tid = TID_1D;
                                if(tid==0)printf("AsyncPushDDPrio\n");  
            if (LB == LoadBalancing::NONE) {
                async_push_dd::Relax(app_inst,
                                     work_source,
                                     work_target_low,
                                     work_target_high,
                                     (TBuffer) current_priority,
                                     csr_graph,
                                     node_value_datum,
                                     node_buffer_datum,
                                     edge_weight_datum);
            } else {
                async_push_dd::RelaxCTA<LB>(app_inst,
                                            work_source,
                                            work_target_low,
                                            work_target_high,
                                            (TBuffer) current_priority,
                                            csr_graph,
                                            node_value_datum,
                                            node_buffer_datum,
                                            edge_weight_datum);
            }
        }

        template<LoadBalancing LB,
                typename TAppInst,
                typename PMAGraph,
                template<typename> class GraphDatum,
                typename TValue,
                typename TBuffer,
                typename TWeight>
        __global__
        void AsyncPushDDFused(TAppInst app_inst,
                              groute::dev::Queue<index_t> queue_input,
                              groute::dev::Queue<index_t> queue_output,
                              PMAGraph csr_graph,
                              GraphDatum<TValue> node_value_datum,
                              GraphDatum<TBuffer> node_buffer_datum,
                              GraphDatum<TWeight> edge_weight_datum,
                              cub::GridBarrier grid_barrier) {
            uint32_t tid = TID_1D;
            groute::dev::Queue<index_t> *p_input = &queue_input;
            groute::dev::Queue<index_t> *p_output = &queue_output;
            if(tid==0)printf("AsyncPushDDFused\n");  
            assert(p_input->count() > 0);
            assert(p_output->count() == 0);

            while (p_input->count()) {
                groute::dev::WorkSourceArray<index_t> work_source(p_input->data_ptr(), p_input->count());

                auto &work_target = *p_output;

                if (LB == LoadBalancing::NONE) {
                    async_push_dd::Relax(app_inst,
                                         work_source,
                                         work_target,
                                         work_target,
                                         (TBuffer) 0,
                                         csr_graph,
                                         node_value_datum,
                                         node_buffer_datum,
                                         edge_weight_datum);
                } else {
                    async_push_dd::RelaxCTA<LB>(app_inst,
                                                work_source,
                                                work_target,
                                                work_target,
                                                (TBuffer) 0,
                                                csr_graph,
                                                node_value_datum,
                                                node_buffer_datum,
                                                edge_weight_datum);
                }

                grid_barrier.Sync(); // this barrier to ensure computation done

                if (tid == 0) {
//                        LOG("Round: %u In: %u Out: %u\n",
//                            *app_inst.m_p_current_round,
//                            work_source.get_size(),
//                            work_target.count());
                    app_inst.PostComputation();
                    *app_inst.m_p_current_round += 1;
                    p_input->reset();
                }

                utils::swap(p_input, p_output);
                grid_barrier.Sync(); // this barrier to ensure reset done
            }

            // fix the last iteration times
            if (tid == 0) {
                *app_inst.m_p_current_round -= 1;
            }
        }

        template<LoadBalancing LB,
                typename TAppInst,
                typename PMAGraph,
                template<typename> class GraphDatum,
                typename TValue,
                typename TBuffer,
                typename TWeight>
        __global__
        void AsyncPushDDFusedPrio(TAppInst app_inst,
                                  groute::dev::Queue<index_t> queue_input,
                                  groute::dev::Queue<index_t> queue_output_low,
                                  groute::dev::Queue<index_t> queue_output_high,
                                  TBuffer current_priority,
                                  PMAGraph csr_graph,
                                  GraphDatum<TValue> node_value_datum,
                                  GraphDatum<TBuffer> node_buffer_datum,
                                  GraphDatum<TWeight> edge_weight_datum,
                                  cub::GridBarrier grid_barrier) {
            const uint32_t tid = TID_1D;
            const TBuffer step = current_priority;
            groute::dev::Queue<index_t> *p_input = &queue_input;
            groute::dev::Queue<index_t> *p_output_low = &queue_output_low;
            groute::dev::Queue<index_t> *p_output_high = &queue_output_high;
            groute::dev::WorkSourceRange<index_t> work_source_all(0, csr_graph.nnodes);

            assert(p_input->count() > 0);
            assert(p_output_low->count() == 0);
            assert(p_output_high->count() == 0);

            while (p_input->count()) {
                while (p_input->count()) {
                    groute::dev::WorkSourceArray<index_t> work_source(p_input->data_ptr(), p_input->count());

                    if (LB == LoadBalancing::NONE) {
                        async_push_dd::Relax(app_inst,
                                             work_source,
                                             *p_output_low,
                                             *p_output_high,
                                             current_priority,
                                             csr_graph,
                                             node_value_datum,
                                             node_buffer_datum,
                                             edge_weight_datum);
                    } else {
                        async_push_dd::RelaxCTA<LB>(app_inst,
                                                    work_source,
                                                    *p_output_low,
                                                    *p_output_high,
                                                    current_priority,
                                                    csr_graph,
                                                    node_value_datum,
                                                    node_buffer_datum,
                                                    edge_weight_datum);
                    }
                    grid_barrier.Sync(); // this barrier to ensure computation done

                    if (tid == 0) {
//                            LOG("Round: %u In: %u Low: %u High: %u Prio: %u\n",
//                                *app_inst.m_p_current_round,
//                                work_source.get_size(),
//                                p_output_low->count(),
//                                p_output_high->count(),
//                                current_priority);
                if(tid==0)printf("AsyncPushDDFusedPrio\n"); 
                        app_inst.PostComputation();
                        *app_inst.m_p_current_round += 1;
                        p_input->reset();
                    }
                    grid_barrier.Sync(); // wait for reset done
                    utils::swap(p_output_high, p_input);
                }
                current_priority += step;

                utils::swap(p_input, p_output_low);
                grid_barrier.Sync(); // this barrier to ensure reset done
            }

            // fix the last iteration times
            if (tid == 0) {
                *app_inst.m_p_current_round -= 1;
            }
        }

        template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename TBitmap,
                typename CSCGraph,
                template<typename> class GraphDatum,
                typename TBuffer,
                typename TWeight>
        __global__
        void AsyncPullDD(TAppInst app_inst,
                         WorkSource work_source,
                         TBitmap in_active,
                         TBitmap out_active,
                         CSCGraph csc_graph,
                         GraphDatum<TBuffer> node_buffer_datum,
                         GraphDatum<TWeight> edge_weight_datum){
            if (LB == LoadBalancing::NONE) {
                async_pull_dd::Relax(app_inst,
                                     work_source,
                                     in_active,
                                     out_active,
                                     out_active,
                                     (TBuffer) 0,
                                     csc_graph,
                                     node_buffer_datum,
                                     edge_weight_datum);
            } else {
                async_pull_dd::RelaxCTA<LB>(app_inst,
                                            work_source,
                                            in_active,
                                            out_active,
                                            out_active,
                                            (TBuffer) 0,
                                            csc_graph,
                                            node_buffer_datum,
                                            edge_weight_datum);
            }
        }

        template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename TBitmap,
                typename CSCGraph,
                template<typename> class GraphDatum,
                typename TBuffer,
                typename TWeight>
        __global__
        void AsyncPullDDPrio(TAppInst app_inst,
                             WorkSource work_source,
                             TBitmap in_active,
                             TBitmap out_active_low,
                             TBitmap out_active_high,
                             TBuffer current_priority,
                             CSCGraph csc_graph,
                             GraphDatum<TBuffer> node_buffer_datum,
                             GraphDatum<TWeight> edge_weight_datum) {
            if (LB == LoadBalancing::NONE) {
                async_pull_dd::Relax(app_inst,
                                     work_source,
                                     in_active,
                                     out_active_low,
                                     out_active_high,
                                     current_priority,
                                     csc_graph,
                                     node_buffer_datum,
                                     edge_weight_datum);
            } else {
                async_pull_dd::RelaxCTA<LB>(app_inst,
                                            work_source,
                                            in_active,
                                            out_active_low,
                                            out_active_high,
                                            current_priority,
                                            csc_graph,
                                            node_buffer_datum,
                                            edge_weight_datum);
            }
        }
        template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename CSCGraph,
                template<typename> class GraphDatum,
                typename TBuffer,
                typename TWeight>
        __global__
        void SyncPullTD(TAppInst app_inst,
			index_t seg_snode,
			index_t seg_sedge_csc,
                        WorkSource work_source,
                        CSCGraph csc_graph,
                        GraphDatum<TBuffer> node_in_buffer_datum,
                        GraphDatum<TBuffer> node_out_buffer_datum,
                        GraphDatum<TWeight> edge_weight_datum) {
                            uint32_t tid = TID_1D;
                            if(tid==0)printf("SyncPullTD\n");  
            if (LB == LoadBalancing::NONE) {
		
                sync_pull_td::Relax(app_inst,seg_snode,seg_sedge_csc,
                                    work_source,
                                    csc_graph,
                                    node_in_buffer_datum,
                                    node_out_buffer_datum,
                                    edge_weight_datum);
            } else {
                sync_pull_td::RelaxCTA<LB>(app_inst,seg_snode,seg_sedge_csc,
                                           work_source,
                                           csc_graph,
                                           node_in_buffer_datum,
                                           node_out_buffer_datum,
                                           edge_weight_datum);
            }
        }        
        template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename TBitmap,
                typename CSCGraph,
                template<typename> class GraphDatum,
                typename TBuffer,
                typename TWeight>
        __global__
        void SyncPullDD(TAppInst app_inst,
			index_t seg_snode,LoadBalancing,
			   index_t seg_enode,
			   index_t seg_sedge_csc,
			bool zcflag,
                        WorkSource work_source,
                        TBitmap in_active,
                        TBitmap out_active,
                        CSCGraph csc_graph,
                        GraphDatum<TBuffer> node_in_buffer_datum,
                        GraphDatum<TBuffer> node_out_buffer_datum,
                        GraphDatum<TWeight> edge_weight_datum) {
                            uint32_t tid = TID_1D;
                            if(tid==0)printf("SyncPullDD\n"); 
            if (LB == LoadBalancing::NONE) {
                sync_pull_dd::Relax(app_inst,seg_snode,seg_enode,seg_sedge_csc,zcflag,
                                    work_source,
                                    in_active,
                                    out_active,
                                    csc_graph,
                                    node_in_buffer_datum,
                                    node_out_buffer_datum,
                                    edge_weight_datum);
            } else {
                sync_pull_dd::RelaxCTA<LB>(app_inst,
                                           work_source,
                                           in_active,
                                           out_active,
                                           csc_graph,
                                           node_in_buffer_datum,
                                           node_out_buffer_datum,
                                           edge_weight_datum);
            }
        }

        template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename TBitmap,
                typename CSCGraph,
                template<typename> class GraphDatum,
                typename TBuffer,
                typename TWeight>
        __global__
        void SyncPullDD_test(TAppInst app_inst,
			index_t seg_snode,
			   index_t seg_enode,
			   index_t seg_sedge_csc,
			bool zcflag,
                        WorkSource work_source,
                        TBitmap in_active,
                        TBitmap out_active,
                        CSCGraph csc_graph,
                        GraphDatum<TBuffer> node_in_buffer_datum,
                        GraphDatum<TBuffer> node_out_buffer_datum,
                        GraphDatum<TWeight> edge_weight_datum) {
                            uint32_t tid = TID_1D;
                            if(tid==0)printf("SyncPullDD\n"); 
            if (zcflag) {
                sync_pull_dd::RelaxCTA_ZC<LB>(app_inst,seg_snode,seg_enode,seg_sedge_csc,
                                    work_source,
                                    in_active,
                                    out_active,
                                    csc_graph,
                                    node_in_buffer_datum,
                                    node_out_buffer_datum,
                                    edge_weight_datum);
            } else {
                sync_pull_dd::RelaxCTA_segment<LB>(app_inst,seg_snode,seg_enode,seg_sedge_csc,
                                           work_source,
                                           in_active,
                                           out_active,
                                           csc_graph,
                                           node_in_buffer_datum,
                                           node_out_buffer_datum,
                                           edge_weight_datum);
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
        __global__
        void SyncPushDD(TAppInst app_inst,
			index_t seg_snode,
			   index_t seg_enode,
			   uint64_t seg_sedge_csr,
			 bool zcflag,
                         WorkSource work_source,
                         WorkTarget work_target,
                         PMAGraph csr_graph,
                         GraphDatum<TValue> node_value_datum,
                         GraphDatum<TBuffer> node_buffer_datum,
                         GraphDatum<TWeight> edge_weight_datum) {
                            uint32_t tid = TID_1D;
              //             if(tid==0)printf("SyncPushDD\n");  
            //if (LB == LoadBalancing::NONE)
            if (true)
             {
                sync_push_dd::Relax(app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                                     work_source,
                                     work_target,
                                     work_target,
                                     (TBuffer) 0,
                                     csr_graph,
                                     node_value_datum,
                                     node_buffer_datum,
                                     edge_weight_datum);
            } else {
                sync_push_dd::RelaxCTA<LB>(app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                                            work_source,
                                            work_target,
                                            work_target,
                                            (TBuffer) 0,
                                            csr_graph,
                                            node_value_datum,
                                            node_buffer_datum,
                                            edge_weight_datum);
            }
        }
         template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename PMAGraph,
                typename Hotness,
                typename BufferVec,
                typename TValue,typename TBuffer>
        __global__
        void SyncPushDDBAll(TAppInst app_inst,
                         index_t seg_snode,
			             index_t seg_enode,
			             uint64_t seg_sedge_csr,
			             bool zcflag,
                         WorkSource work_source,
                         const PMAGraph vcsr_graph,
                         Hotness hotness,
                         BufferVec buffer,
                         BufferVec buffer_l2,
                        uint64_t *count_gpu,
                         uint64_t *total_act_d,
                         uint64_t* cache_size,
                         TValue* node_value_datum,
                         TValue* node_parent_datum,
                         TBuffer* node_buffer_datum,
                         BitmapDeviceObject out_active,
                         BitmapDeviceObject in_active) {
                sync_push_dd::RelaxCTADB_all_vertices<LB, false>(app_inst,seg_snode,seg_enode, seg_sedge_csr,zcflag,
                                                   work_source,
                                                   vcsr_graph,
                                                   hotness,
                                                   buffer,
                                                   buffer_l2,
                                                   count_gpu,
                                                   total_act_d,
                                                   cache_size,
                                                   node_value_datum,
                                                   node_parent_datum,
                                                   node_buffer_datum,
                                                   (TBuffer) 0,
                                                   out_active,
                                                   in_active);
        }

         template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename PMAGraph,
                typename BufferVec,
                typename TValue,
                typename TBuffer>
        __global__
        void SyncPushDDBAmend(TAppInst app_inst,
                         index_t seg_snode,
			             index_t seg_enode,
			             uint64_t seg_sedge_csr,
			             bool zcflag,
                         WorkSource work_source,
                         const PMAGraph vcsr_graph,
                         int *type_device,
                         BufferVec *buffer,
                         BufferVec *buffer_l2,
                         uint64_t *total_act_d,
                         uint64_t *cache_size,
                        TValue *node_parent_datum,
                        TValue *node_value_datum,
                        TBuffer * node_buffer_datum,
                        // TBuffer * node_tmp_buffer_datum,
                         BitmapDeviceObject out_active,
                         BitmapDeviceObject in_active) {
                sync_push_dd::RelaxCTADBAmend<LB, false>(app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                                                   work_source,
                                                   vcsr_graph,
                                                   type_device,
                                                   buffer,
                                                   buffer_l2,
                                                   total_act_d,
                                                   cache_size,
                                                   node_parent_datum,
                                                   node_value_datum,
                                                   node_buffer_datum,
                                                //    node_tmp_buffer_datum,
                                                   (TBuffer) 0,
                                                   out_active,
                                                   in_active);
        }
         template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename PMAGraph,
                typename BufferVec,
                typename TValue,
                typename TBuffer>
        __global__
        void SyncPushDDBCache(TAppInst app_inst,
                         index_t seg_snode,
			             index_t seg_enode,
			             uint64_t seg_sedge_csr,
			             bool zcflag,
                         WorkSource work_source,
                         const PMAGraph vcsr_graph,
                         int *type_device,
                         BufferVec *buffer,
                         BufferVec *buffer_l2,
                         uint64_t *cache_size,
                        TValue *node_parent_datum,
                        TValue *node_value_datum,
                        TBuffer * node_buffer_datum,
                        // TBuffer * node_tmp_buffer_datum,
                         BitmapDeviceObject out_active,
                         BitmapDeviceObject in_active) {
                sync_push_dd::RelaxCTADBCache<LB, false>(app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                                                   work_source,
                                                   vcsr_graph,
                                                   type_device,
                                                   buffer,
                                                   buffer_l2,
                                                   cache_size,
                                                   node_parent_datum,
                                                   node_value_datum,
                                                   node_buffer_datum,
                                                //    node_tmp_buffer_datum,
                                                   (TBuffer) 0,
                                                   out_active,
                                                   in_active);
        }
         template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename PMAGraph,
                typename BufferVec,
                typename TValue,
                typename TBuffer>
        __global__
        void SyncPushDDBFlush(TAppInst app_inst,
                         index_t seg_snode,
			             index_t seg_enode,
			             uint64_t seg_sedge_csr,
			             bool zcflag,
                         WorkSource work_source,
                         const PMAGraph vcsr_graph,
                         BufferVec *buffer,
                         BufferVec *buffer_l2,
                         BufferVec *buffer_l3,
                         uint64_t *cache_size,
                         TValue* node_value_datum,
                         TValue* node_parent_datum,
                         TBuffer* node_buffer_datum,
                         BitmapDeviceObject out_active,
                         BitmapDeviceObject in_active) {
                sync_push_dd::RelaxCTADBFlush<LB, false>(app_inst,seg_snode,seg_enode, seg_sedge_csr,zcflag,
                                                   work_source,
                                                   vcsr_graph,
                                                   buffer,
                                                   buffer_l2,
                                                   buffer_l3,
                                                   cache_size,
                                                   node_value_datum,
                                                   node_parent_datum,
                                                   node_buffer_datum,
                                                   (TBuffer) 0,
                                                   out_active,
                                                   in_active);
        }

         template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename PMAGraph,
                typename Hot,
                typename BufferVec,
                typename TValue,
                typename TBuffer>
        __global__
        void SyncPushDDBDelta(TAppInst app_inst,
                         index_t seg_snode,
			             index_t seg_enode,
			             uint64_t seg_sedge_csr,
			             bool zcflag,
                         WorkSource work_source,
                         const PMAGraph vcsr_graph,
                         Hot hotness,
                         BufferVec *buffer,
                         BufferVec *buffer_l2,
                        uint64_t *count_gpu,
                         uint64_t *total_act_d,
                         uint64_t *cache_size,
                        TValue *node_parent_datum,
                        TValue *node_value_datum,
                        TBuffer * node_buffer_datum,
                        BitmapDeviceObject out_active,
                        BitmapDeviceObject in_active) {
                sync_push_dd::RelaxCTADB<LB, false>(app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                                                   work_source,
                                                   vcsr_graph,
                                                   hotness,
                                                   buffer,
                                                   buffer_l2,
                                                   count_gpu,
                                                   total_act_d,
                                                   cache_size,
                                                   node_parent_datum,
                                                   node_value_datum,
                                                   node_buffer_datum,
                                                   (TBuffer) 0,
                                                   out_active,
                                                   in_active);
        }
        
         template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename PMAGraph,
                typename BufferVec,
                typename HashNodes,
                // template<typename> class GraphDatum,
                typename TValue,
                typename TBuffer>
        __global__
        void SyncPushDDB(TAppInst app_inst,
                         index_t seg_snode,
			             index_t seg_enode,
			             uint64_t seg_sedge_csr,
			             bool zcflag,
                         WorkSource work_source,
                         const PMAGraph vcsr_graph,
                         BufferVec buffer,
                         BufferVec buffer_l2,
                         HashNodes nodes_hash,
                        // groute::graphs::dev::PMAGraph vcsr_graph,
                        TValue *node_parent_datum,
                        TValue *node_value_datum,
                        TBuffer * node_buffer_datum,
                        // TValue *node_level_datum,
                        // GraphDatum<TWeight> edge_weight_datum,
                        BitmapDeviceObject out_active,
                        BitmapDeviceObject in_active) {
                sync_push_dd::RelaxCTADB<LB, false>(app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                                                   work_source,
                                                   vcsr_graph,
                                                   buffer,
                                                   buffer_l2,
                                                   nodes_hash,
                                                   node_parent_datum,
                                                   node_value_datum,
                                                   node_buffer_datum,
                                                //    node_level_datum,
                                                //    edge_weight_datum,
                                                   (TBuffer) 0,
                                                   out_active,
                                                   in_active);
        }

/**
 * @brief Quaful SyncPushAdd kernel function
 * @tparam TAppInst  Type of the application
 * @tparam TPMAGraph  Type of graph
 * @tparam TGraphDatum  Type of device graph associate data
//  */
         template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename PMAGraph,
                typename BufferVec,
                typename TValue,
                typename TBuffer>
        __global__
        void SyncPushDDB_del(TAppInst app_inst,
                         index_t seg_snode,
			             index_t seg_enode,
			             uint64_t seg_sedge_csr,
			             bool zcflag,
                         WorkSource work_source,
                         const PMAGraph vcsr_graph,
                         BufferVec buffer,
                         BufferVec buffer_l2,
                         uint64_t *count_gpu,
                         uint64_t *total_act_d,
                         bool* reset_node,
                        TValue *node_parent_datum,
                        TValue *node_value_datum,
                        TBuffer * node_buffer_datum,
                        BitmapDeviceObject out_active,
                        BitmapDeviceObject in_active) {
                sync_push_dd::RelaxCTADB_del<LB, false>(app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                                                   work_source,
                                                   vcsr_graph,
                                                   buffer,
                                                   buffer_l2,
                                                   count_gpu,
                                                   total_act_d,
                                                   reset_node,
                                                   node_parent_datum,
                                                   node_value_datum,
                                                   node_buffer_datum,
                                                   (TBuffer) 0,
                                                   out_active,
                                                   in_active);
        }
        
/**
 * @brief Quaful SyncPushAdd kernel function
 * @tparam TAppInst  Type of the application
 * @tparam TPMAGraph  Type of graph
 * @tparam TGraphDatum  Type of device graph associate data
 */
         template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename PMAGraph,
                typename TVec,
                typename TValue,
                typename TBuffer>
        __global__
        void SyncPushDDB_add(TAppInst app_inst,
                         index_t seg_snode,
			             index_t seg_enode,
			             uint64_t seg_sedge_csr,
			             bool zcflag,
                         WorkSource work_source,
                         const PMAGraph vcsr_graph,
                        TValue *node_parent_datum,
                        TVec cache_edges_l2,
                        uint64_t *cache_size,
                        TValue *node_value_datum,
                        TBuffer * node_buffer_datum,
                        BitmapDeviceObject out_active,
                        BitmapDeviceObject in_active) {
                sync_push_dd::RelaxCTADBAdd<LB, false>(app_inst,seg_snode,seg_enode,seg_sedge_csr,zcflag,
                                                   work_source,
                                                   vcsr_graph,
                                                   node_parent_datum,
                                                   cache_edges_l2,
                                                   cache_size,
                                                   node_value_datum,
                                                   node_buffer_datum,
                                                   (TBuffer) 0,
                                                   out_active,
                                                   in_active);
        }

                 template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename PMAGraph,
                template<typename> class GraphDatum,
                typename TValue,
                typename TBuffer,
                typename TWeight>
        __global__
        void SyncPushDDB_COM(TAppInst app_inst,
                         WorkSource work_source,
                         const PMAGraph csr_graph,
                        //  GraphDatum<TValue> node_value_datum,
                        //  GraphDatum<TBuffer> node_buffer_datum,
                        TValue *node_value_datum,
                        TBuffer *node_buffer_datum,
                        TValue *node_parent_datum,
                         GraphDatum<TWeight> edge_weight_datum,
                         BitmapDeviceObject out_active,
                         BitmapDeviceObject in_active) {

                sync_push_dd::RelaxCTADB_COM<LB, false>(app_inst,
                                                   work_source,
                                                   csr_graph,
                                                   node_value_datum,
                                                   node_buffer_datum,
                                                   node_parent_datum,
                                                   edge_weight_datum,
                                                   (TBuffer) 0,
                                                   out_active,
                                                   in_active);
            
        }


        template<LoadBalancing LB,
                typename TAppInst,
                typename WorkSource,
                typename CSCGraph,
                template<typename> class GraphDatum,
                typename TBuffer,
                typename TWeight>
        __global__
        void SyncPullTDB(TAppInst app_inst,
			index_t seg_snode,
			index_t seg_sedge_csc,
			 bool zcflag,
                        WorkSource work_source,
                        CSCGraph csc_graph,
                        GraphDatum<TBuffer> node_in_buffer_datum,
                        GraphDatum<TBuffer> node_out_buffer_datum,
                        GraphDatum<TWeight> edge_weight_datum,
			BitmapDeviceObject out_active,
                         BitmapDeviceObject in_active) {
                            uint32_t tid = TID_1D;
                            //if(tid==0)printf("SyncPullTD\n");  
                sync_pull_td::RelaxCTADB<LB>(app_inst,seg_snode,seg_sedge_csc,zcflag,
                                           work_source,
                                           csc_graph,
                                           node_in_buffer_datum,
                                           node_out_buffer_datum,
                                           edge_weight_datum,
					   out_active,
					   in_active
					    );
            
        }  


    }
}
#endif //HYBRID_DRIVER_H
