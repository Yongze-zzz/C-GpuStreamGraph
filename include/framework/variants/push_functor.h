// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_PUSH_FUNCTOR_H
#define HYBRID_PUSH_FUNCTOR_H

#include <groute/device/queue.cuh>
#include <groute/device/bitmap_impls.h>
#include <framework/variants/api.cuh>

namespace sepgraph
{
    namespace kernel
    {
        template <typename TBuffer>
        struct Payload
        {
            index_t m_src;
            TBuffer m_buffer_to_push;
            //https://stackoverflow.com/questions/33978185/default-constructor-cannot-be-referenced-in-visual-studio-2015
            //X is a union-like class that has a variant member with a non-trivial default constructor,
            //            __device__
            //            Payload()
            //            {
            //
            //            }
            //
            //            __device__ __forceinline__
            //            Payload(index_t src,
            //                    TBuffer buffer_to_push) : m_src(src),
            //                                              m_buffer_to_push(buffer_to_push)
            //            {
            //
            //            }
        };

        template <typename TAppInst,
                  typename PMAGraph,
                  template <typename> class GraphDatum,
                  typename TBuffer,
                  typename TWeight>
        struct PushFunctor
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            PMAGraph m_vcsr_graph;
            GraphDatum<TBuffer> m_buffer_array;
            GraphDatum<TWeight> m_weight_array;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;

            __device__
            PushFunctor()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param csr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctor(TAppInst app_inst,
                        PMAGraph csr_graph,
                        GraphDatum<TBuffer> buffer_array,
                        GraphDatum<TWeight> weight_array) : m_app_inst(app_inst),
                                                            m_work_target_low(nullptr, nullptr, 0),
                                                            m_work_target_high(nullptr, nullptr, 0),
                                                            m_current_priority(0),
                                                            m_vcsr_graph(csr_graph),
                                                            m_buffer_array(buffer_array),
                                                            m_weight_array(weight_array),
                                                            m_data_driven(false),
                                                            m_priority(false)
            {
                m_weighted = m_weight_array.size > 0;
            }

            /**
             * Async+Push+DD+[Priority]
             * @param app_inst
             * @param work_target_low
             * @param work_target_high
             * @param current_priority
             * @param csr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctor(TAppInst app_inst,
                        TWorkTarget work_target_low,
                        TWorkTarget work_target_high,
                        TBuffer current_priority,
                        PMAGraph csr_graph,
                        GraphDatum<TBuffer> buffer_array,
                        GraphDatum<TWeight> weight_array) : m_app_inst(app_inst),
                                                            m_work_target_low(work_target_low),
                                                            m_work_target_high(work_target_high),
                                                            m_current_priority(current_priority),
                                                            m_vcsr_graph(csr_graph),
                                                            m_buffer_array(buffer_array),
                                                            m_weight_array(weight_array),
                                                            m_data_driven(true)
            {
                m_weighted = m_weight_array.size > 0;
                m_priority = work_target_low != work_target_high;
            }

            __device__ __forceinline__ bool operator()(index_t edge, Payload<TBuffer> meta_data)
            {
                // index_t dst = m_vcsr_graph.edge_dest(edge);
                index_t dst = m_vcsr_graph.edges_[edge];
                TBuffer buffer_to_push = meta_data.m_buffer_to_push;
                int status;
                bool accumulate_success;
                bool continue_push;

                if (m_weighted)
                {
                    status = m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                         m_weight_array[edge],
                                                         m_buffer_array.get_item_ptr(dst),
                                                         buffer_to_push);
                }
                else
                {
                    status = m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                         m_buffer_array.get_item_ptr(dst),
                                                         buffer_to_push);
                }

                continue_push = (status == m_app_inst.ACCUMULATE_SUCCESS_CONTINUE ||
                                 status == m_app_inst.ACCUMULATE_FAILURE_CONTINUE);
                accumulate_success = (status == m_app_inst.ACCUMULATE_SUCCESS_BREAK ||
                                      status == m_app_inst.ACCUMULATE_SUCCESS_CONTINUE);

                if (m_data_driven && accumulate_success)
                {
                    if (m_priority)
                    {
                        if (m_app_inst.IsHighPriority(m_current_priority, m_buffer_array[dst]))
                            m_work_target_high.append(dst);
                        else
                            m_work_target_low.append(dst);
                    }
                    else
                    {
                        m_work_target_high.append(dst);
                    }
                }

                return continue_push;
            }
        };
        template <typename TAppInst,
                  typename TPMAGraph,
                  typename Buffer,
                  typename TValue,
                  typename TBuffer>
        struct PushFunctorDBAmend_cache
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            TPMAGraph m_vcsr_graph;
            Buffer *m_cache_g;
            uint64_t *m_cache_size;
            TValue *m_parent_array;
            TBuffer *m_buffer_array;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDBAmend_cache()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param vcsr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDBAmend_cache(TAppInst app_inst,
            const TPMAGraph vcsr_graph,
            Buffer *buffer,
            uint64_t *cache_size,
            TValue *parent_array,
            TBuffer *buffer_array,
            BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                            m_work_target_low(nullptr, nullptr, 0),
                                            m_work_target_high(nullptr, nullptr, 0),
                                            m_current_priority(0),
                                            m_vcsr_graph(vcsr_graph),
                                            m_cache_g(buffer),
                                            m_cache_size(cache_size),
                                            m_parent_array(parent_array),
                                            m_buffer_array(buffer_array),
                                            m_data_driven(true),
                                            m_priority(false),
                                            m_out_active_high(out_active)
            {
                m_weighted = true;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = m_vcsr_graph.edge_dest(edge);
                index_t weight = (meta_data.m_src + dst)%128 + 1;
                // if(m_vcsr_graph.vertices_[meta_data.m_src].cache){
                    uint64_t offset = edge - m_vcsr_graph.begin_edge(meta_data.m_src);

                    index_t here = m_vcsr_graph.vertices_[meta_data.m_src].virtual_start + offset;
                    atomicExch((m_cache_g+here),dst);
                    // m_cache_g[here]=dst;
                    atomicAdd(&m_vcsr_graph.vertices_[meta_data.m_src].virtual_degree,1);
                    // printf("need cache node %d ,dst %d\n",meta_data.m_src,m_cache_g[here]);
                    // m_vcsr_graph.vertices_[meta_data.m_src].virtual_degree++;
                    // if(meta_data.m_src==0){
                    //     printf("src %d dst %d offset %d vir_start %d here %d dev_dst %d\n",
                    //     meta_data.m_src,dst,offset,m_vcsr_graph.vertices_[meta_data.m_src].virtual_start,here,m_cache_g[here]);
                    // }
                // }
                // uint64_t offset = edge - m_vcsr_graph.begin_edge(meta_data.m_src);
                // index_t here = m_vcsr_graph.vertices_[meta_data.m_src].virtual_start + offset;
                // atomicExch((m_cache_g+here),dst);
                // atomicAdd(&m_vcsr_graph.vertices_[meta_data.m_src].virtual_degree,1);

                return true;
            }
        };
        template <typename TAppInst,
                  typename TPMAGraph,
                  typename Buffer,
                  typename TValue,
                  typename TBuffer>
        struct PushFunctorDB
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            TPMAGraph m_vcsr_graph;
            Buffer *m_cache_g;
            uint64_t *m_cache_size;
            TValue *m_parent_array;
            TBuffer *m_buffer_array;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDB()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param vcsr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDB(TAppInst app_inst,
            const TPMAGraph vcsr_graph,
            Buffer *buffer,
            uint64_t *cache_size,
            TValue *parent_array,
            TBuffer *buffer_array,
            BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                            m_work_target_low(nullptr, nullptr, 0),
                                            m_work_target_high(nullptr, nullptr, 0),
                                            m_current_priority(0),
                                            m_vcsr_graph(vcsr_graph),
                                            m_cache_g(buffer),
                                            m_cache_size(cache_size),
                                            m_parent_array(parent_array),
                                            m_buffer_array(buffer_array),
                                            m_data_driven(true),
                                            m_priority(false),
                                            m_out_active_high(out_active)
            {
                m_weighted = true;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = m_vcsr_graph.edge_dest(edge);
                index_t weight = (meta_data.m_src + dst)%128 + 1;
                
                TBuffer buffer_to_push = meta_data.m_buffer_to_push;
                if ((dst!=-1))
                {  
                        m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                        weight,
                                                        &m_parent_array[dst],
                                                        &m_buffer_array[dst],
                                                        buffer_to_push);
                }

                return true;
            }
        };

        template <typename TAppInst,
                  typename PMAGraph,
                  template <typename> class GraphDatum,
                  typename TValue,
                  typename TBuffer,
                  typename TWeight>
        struct PushFunctorDB_COM
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            PMAGraph m_vcsr_graph;
            TValue *m_parent_array;
            TBuffer *m_buffer_array;
            GraphDatum<TWeight> m_weight_array;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDB_COM()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param csr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDB_COM(TAppInst app_inst,
                          PMAGraph csr_graph,
                          TValue *parent_array,
                          TBuffer *buffer_array,
                          GraphDatum<TWeight> weight_array,
                          BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(csr_graph),
                                                           m_parent_array(parent_array),
                                                           m_buffer_array(buffer_array),
                                                           m_weight_array(weight_array),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_out_active_high(out_active)
            {
                m_weighted = m_weight_array.size > 0;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                // index_t dst =  m_vcsr_graph.edge_dest(edge);
                index_t dst =  m_vcsr_graph.edge_dest(edge);
                TBuffer buffer_to_push = meta_data.m_buffer_to_push;

                // if (m_weighted)
                // {
                    m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                         m_weight_array[edge],
                                                          &m_parent_array[dst],
                                                         &m_buffer_array[dst],
                                                         buffer_to_push);
                // }
                // else
                // {
            
                //    m_app_inst.AccumulateBuffer(meta_data.m_src,
                //                                          dst,
                //                                          m_buffer_array[dst],
                //                                          buffer_to_push);
                // }


                return true;
            }
        };

        template <typename TAppInst,
                  typename TPMAGraph,
                //   template <typename> class GraphDatum,
                  typename TValue,
                  typename TBuffer,
                  typename TVec>
        struct PushFunctorAdd
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            TPMAGraph m_vcsr_graph;
            TValue *m_parent_array;
            uint64_t *m_cache_size;
            TBuffer *m_buffer_array;
            TValue *m_value_array;
            TVec m_cache_g;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorAdd()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param vcsr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorAdd(TAppInst app_inst,
            const TPMAGraph vcsr_graph,
                        TValue *parent_array,
                        TBuffer *buffe_array,
                        TVec cache_l2,
                        uint64_t *cache_size,
                        BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(vcsr_graph),
                                                           m_parent_array(parent_array),
                                                           m_buffer_array(buffe_array),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_cache_g(cache_l2),
                                                           m_cache_size(cache_size),
                                                           m_out_active_high(out_active)
            {
                m_weighted = true;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = m_vcsr_graph.edge_dest(edge);
                // if(meta_data.m_src == 32) printf("========= src %d dst %d\n",meta_data.m_src,dst);
                uint64_t offset = edge - m_vcsr_graph.begin_edge(meta_data.m_src) -m_vcsr_graph.vertices_[meta_data.m_src].virtual_degree; 
                // uint64_t offset = 0;
                // uint64_t here = m_vcsr_graph.vertices_[meta_data.m_src].secondary_start + offset;
                // TBuffer buffer_to_push = meta_data.m_buffer_to_push;
                if((m_vcsr_graph.vertices_[meta_data.m_src].secondary_start + offset) < ((*m_cache_size)/2)){
                    uint64_t here = m_vcsr_graph.vertices_[meta_data.m_src].secondary_start + offset;
                    m_cache_g[here] = dst;
                    atomicAdd(&(m_vcsr_graph.vertices_[meta_data.m_src].secondary_degree),1);
                    // LOG("cache L-2 location: %lu edges %d %lu\n",
                    // here,
                    // meta_data.m_src,
                    // m_cache_g[here]);
                }
                return true;
            }
        };

        template <typename TAppInst,
                  typename TPMAGraph,
                  typename Buffer,
                  typename TValue,
                  typename TBuffer>
        struct PushFunctorDBADD
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            TPMAGraph m_vcsr_graph;
            Buffer m_cache_g;
            TValue* m_parent_datum;
            TBuffer* m_buffer_datum;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDBADD()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param vcsr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDBADD(TAppInst app_inst,
                            TPMAGraph vcsr_graph,
                            Buffer buffer,
                            TValue* parent_array,
                            TBuffer* buffer_array,
                            BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(vcsr_graph),
                                                           m_cache_g(buffer),
                                                           m_parent_datum(parent_array),
                                                           m_buffer_datum(buffer_array),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_out_active_high(out_active)
            {
                // printf("Construct PushFunctorDB\n");
                m_weighted = true;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = m_vcsr_graph.edge_dest(edge);
                // printf("all 1 src %d dst %d\n",meta_data.m_src,dst);
                index_t weight = (meta_data.m_src + dst)%128+1;
                TBuffer buffer_to_push = meta_data.m_buffer_to_push;
                // TBuffer buffer_to_push = m_buffer_datum[meta_data.m_src];
                if (dst!=UINT32_MAX)
                {  
                    //here need correct
                        m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                        weight,
                                                        &m_parent_datum[dst],
                                                        &m_buffer_datum[dst],
                                                        buffer_to_push);
                         //insert edges to buffer
                }

                return true;
            }
        };

        template <typename TAppInst,
                  typename TPMAGraph,
                  typename Buffer,
                  typename TBuffer>
        struct PushFunctorDBCacheFlush
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            // groute::graphs::dev::PMAGraph m_vcsr_graph;
            TPMAGraph m_vcsr_graph;
            Buffer *m_cache_g;
            Buffer *m_cache_g_l3;
            uint64_t *m_cache_size;
            int m_cache_level;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDBCacheFlush()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param vcsr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDBCacheFlush(TAppInst app_inst,
            const TPMAGraph vcsr_graph,
            Buffer *buffer,
            Buffer *buffer_l3,
            uint64_t *cache_size,
            int &cache_level,
            BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(vcsr_graph),
                                                           m_cache_g(buffer),
                                                           m_cache_g_l3(buffer_l3),
                                                            m_cache_size(cache_size),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_cache_level(cache_level),
                                                           m_out_active_high(out_active)
            {
                m_weighted = true;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = (uint32_t)m_cache_g[edge];
                uint64_t offset;
                offset = edge - m_vcsr_graph.vertices_[meta_data.m_src].virtual_start;
                uint64_t location = offset + m_vcsr_graph.vertices_[meta_data.m_src].third_start;
                atomicExch((m_cache_g_l3+location),dst);
                atomicAdd(&m_vcsr_graph.vertices_[meta_data.m_src].third_degree,1);
                return true;
            }
        };

        template <typename TAppInst,
                  typename TPMAGraph,
                //   template <typename> class GraphDatum,
                  typename Buffer,
                  typename TValue,
                  typename TBuffer>
        struct PushFunctorDBCachel2_ADD
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            // groute::graphs::dev::PMAGraph m_vcsr_graph;
            TPMAGraph m_vcsr_graph;
            Buffer m_cache_g;
            TValue* m_parent_datum;
            TBuffer* m_buffer_datum;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDBCachel2_ADD()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param vcsr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDBCachel2_ADD(TAppInst app_inst,
            const TPMAGraph vcsr_graph,
            Buffer buffer,
            TValue* parent_array,
            TBuffer* buffer_array,
            BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(vcsr_graph),
                                                           m_cache_g(buffer),
                                                           m_parent_datum(parent_array),
                                                           m_buffer_datum(buffer_array),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_out_active_high(out_active)
            {
                m_weighted = true;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = (uint32_t)m_cache_g[edge];
                // printf("all 2 src %d dst %d\n",meta_data.m_src,dst);
                index_t weight = (meta_data.m_src + dst) % 128 +1;
                TBuffer buffer_to_push = meta_data.m_buffer_to_push;
                if (dst!=UINT32_MAX)
                {  
                        // m_weight_array[edge] = (meta_data.m_src + dst) % 128;
                        m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                        // m_level_array[meta_data.m_src],
                                                        // &m_level_array[dst],
                                                        weight,
                                                        //  m_buffer_array.get_item_ptr(dst),
                                                        &m_parent_datum[dst],
                                                        &m_buffer_datum[dst],
                                                        buffer_to_push);
                         //insert edges to buffer
                }

                return true;
            }

        };
        template <typename TAppInst,
                  typename TPMAGraph,
                  typename Buffer,
                  typename NodesHash,
                //   template <typename> class GraphDatum,
                  typename TValue,
                  typename TBuffer>
        struct PushFunctorDBCache
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            // groute::graphs::dev::PMAGraph m_vcsr_graph;
            TPMAGraph m_vcsr_graph;
            Buffer m_cache_g;
            NodesHash m_nodes_hash;
            // GraphDatum<TBuffer> m_buffer_array;
            TValue *m_parent_array;
            // TValue *m_level_array;
            TBuffer *m_buffer_array;
            // bool* m_delta_array;
            // GraphDatum<TWeight> m_weight_array;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDBCache()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param vcsr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDBCache(TAppInst app_inst,
            const TPMAGraph vcsr_graph,
            // bool* delta_array,
            Buffer buffer,
            NodesHash nodes_hash,
                        //   groute::graphs::dev::PMAGraph vcsr_graph,
                        TValue *parent_array,
                          TBuffer *buffer_array,
                        //   TValue *level_array,
                        //   GraphDatum<TWeight> weight_array,
                          BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(vcsr_graph),
                                                        //    m_delta_array(delta_array),
                                                           m_cache_g(buffer),
                                                           m_nodes_hash(nodes_hash),
                                                           m_parent_array(parent_array),
                                                           m_buffer_array(buffer_array),
                                                        //    m_level_array(level_array),
                                                        //    m_weight_array(weight_array),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_out_active_high(out_active)
            {
                // m_weighted = true;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {

                // index_t dst = m_vcsr_graph.edge_dest(edge);
                index_t dst = (uint32_t)m_cache_g[edge];
                index_t weight = (meta_data.m_src + dst) % 128 + 1;  
                TBuffer buffer_to_push = meta_data.m_buffer_to_push;
                // if (m_weighted)
                // {  
                        // m_weight_array[edge] = (meta_data.m_src + dst) % 128;
                        m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                        // m_level_array[meta_data.m_src],
                                                        // &m_level_array[dst],
                                                        weight,
                                                        //  m_buffer_array.get_item_ptr(dst),
                                                        &m_parent_array[dst],
                                                        &m_buffer_array[dst],
                                                        buffer_to_push);
                //          //insert edges to buffer
                // }
                // else
                // {
                //     m_app_inst.AccumulateBuffer(meta_data.m_src,
                //                                             dst,
                //                                             // m_buffer_array.get_item_ptr(dst),
                //                                             &m_parent_array[dst],
                //                                             &m_buffer_array[dst],
                //                                             buffer_to_push);
                // }

                return true;
            }

        };

        template <typename TAppInst,
                  typename TPMAGraph,
                  typename Buffer,
                  typename TValue,
                  typename TBuffer>
        struct PushFunctorDBCachel2
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            // groute::graphs::dev::PMAGraph m_vcsr_graph;
            TPMAGraph m_vcsr_graph;
            Buffer *m_cache_g;
            TValue *m_parent_array;
            TBuffer *m_buffer_array;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDBCachel2()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param vcsr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDBCachel2(TAppInst app_inst,
            const TPMAGraph vcsr_graph,
            // bool* delta_array,
            Buffer *buffer,
                        TValue *parent_array,
                          TBuffer *buffer_array,
                          BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(vcsr_graph),
                                                           m_cache_g(buffer),
                                                           m_parent_array(parent_array),
                                                           m_buffer_array(buffer_array),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_out_active_high(out_active)
            {
                // m_weighted = true;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = m_cache_g[edge];
                // printf("have cache %d %d\n",meta_data.m_src,dst);
                index_t weight = (meta_data.m_src + dst) % 128 + 1;
                TBuffer buffer_to_push = meta_data.m_buffer_to_push;
                if (dst!=-1)
                {  
                        // if(dst ==5 )printf("acc src %d -> dst %d delta %f\n",meta_data.m_src,dst,buffer_to_push);
                        m_app_inst.AccumulateBuffer(meta_data.m_src,
                                                         dst,
                                                        weight,
                                                        &m_parent_array[dst],
                                                        &m_buffer_array[dst],
                                                        buffer_to_push);
                         //insert edges to buffer
                }
                // else
                // {
                //     m_app_inst.AccumulateBuffer(meta_data.m_src,
                //                                             dst,
                //                                             // m_buffer_array.get_item_ptr(dst),
                //                                             &m_parent_array[dst],
                //                                             &m_buffer_array[dst],
                //                                             buffer_to_push);
                // }

                return true;
            }

        };

        template <typename TAppInst,
                    typename PMAGraph,
                    typename TVec,
                    typename TValue,
                    typename TBuffer>
        struct PushFunctorDel_l1
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            PMAGraph m_vcsr_graph;
            TValue *m_parent_array;
            // TValue *m_value_array;
            TBuffer *m_buffer_array;
            // GraphDatum<TWeight> m_weight_array;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            TVec m_cache_g;
            // BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDel_l1()
            {
            }
            __device__ __forceinline__
            PushFunctorDel_l1(TAppInst app_inst,
            const PMAGraph vcsr_graph,
                        TVec cache_l1,
                        TValue *parent_array,
                        // TValue *value_array,
                          TBuffer *buffer_array) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(vcsr_graph),
                                                           m_cache_g(cache_l1),
                                                           m_parent_array(parent_array),
                                                           m_buffer_array(buffer_array),
                                                        //    m_weight_array(weight_array),
                                                           m_data_driven(true),
                                                           m_priority(false)
            {
                m_weighted = true;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = (uint32_t)m_cache_g[edge];
                index_t dst_to_del = meta_data.m_buffer_to_push;
                if(dst == dst_to_del) m_cache_g[edge] = UINT32_MAX;
                return true;
            }
        };

        template <typename TAppInst,
                  typename TPMAGraph,
                  typename Buffer,
                  typename TValue,
                  typename TBuffer>
        struct PushFunctorDB_Del
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            // groute::graphs::dev::PMAGraph m_vcsr_graph;
            TPMAGraph m_vcsr_graph;
            Buffer m_cache_g;
            TValue *m_parent_array;
            TValue *m_value_array;
            TBuffer *m_buffer_array;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDB_Del()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param vcsr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDB_Del(TAppInst app_inst,
            const TPMAGraph vcsr_graph,
            Buffer buffer,
                        TValue *parent_array,
                          TBuffer *buffer_array,
                          TValue *value_array,
                          BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(vcsr_graph),
                                                           m_cache_g(buffer),
                                                           m_parent_array(parent_array),
                                                           m_buffer_array(buffer_array),
                                                           m_value_array(value_array),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_out_active_high(out_active)
            {
                m_weighted = true;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = m_vcsr_graph.edge_dest(edge);
            
                TBuffer buffer_to_push = meta_data.m_buffer_to_push;
                if (dst!=UINT32_MAX)
                {  
                        m_app_inst.AccumulateBuffer_del(meta_data.m_src,
                                                         dst,
                                                        &m_parent_array[dst],
                                                        &m_buffer_array[dst],
                                                        &m_value_array[dst]);
                         //insert edges to buffer
                }

                return true;
            }
        };
        template <typename TAppInst,
                  typename TPMAGraph,
                  typename Buffer,
                  typename TValue,
                  typename TBuffer>
        struct PushFunctorDBCachel2_DEL
        {
            typedef groute::dev::Queue<index_t> TWorkTarget;
            TAppInst m_app_inst;
            TWorkTarget m_work_target_low;
            TWorkTarget m_work_target_high;
            // groute::graphs::dev::PMAGraph m_vcsr_graph;
            TPMAGraph m_vcsr_graph;
            Buffer m_cache_g;
            TValue *m_parent_array;
            TValue *m_value_array;
            TBuffer *m_buffer_array;
            TBuffer m_current_priority;
            bool m_data_driven;
            bool m_priority;
            bool m_weighted;
            BitmapDeviceObject m_out_active_high;

            __device__
            PushFunctorDBCachel2_DEL()
            {
            }
            /**
             * Async+Push+TD
             * @param app_inst
             * @param vcsr_graph
             * @param buffer_array
             * @param weight_array
             */
            __device__ __forceinline__
            PushFunctorDBCachel2_DEL(TAppInst app_inst,
            const TPMAGraph vcsr_graph,
            Buffer buffer,
                        TValue *parent_array,
                        TBuffer *buffer_array,
                        TValue *value_array,
                          BitmapDeviceObject out_active) : m_app_inst(app_inst),
                                                           m_work_target_low(nullptr, nullptr, 0),
                                                           m_work_target_high(nullptr, nullptr, 0),
                                                           m_current_priority(0),
                                                           m_vcsr_graph(vcsr_graph),
                                                        //    m_delta_array(delta_array),
                                                           m_cache_g(buffer),
                                                           m_parent_array(parent_array),
                                                           m_buffer_array(buffer_array),
                                                           m_value_array(value_array),
                                                        //    m_weight_array(weight_array),
                                                           m_data_driven(true),
                                                           m_priority(false),
                                                           m_out_active_high(out_active)
            {
                m_weighted = true;
            }

            __device__ __forceinline__ bool operator()(uint64_t edge, Payload<TBuffer> meta_data)
            {
                index_t dst = (uint32_t)m_cache_g[edge];
                TBuffer buffer_to_push = meta_data.m_buffer_to_push;
                if (dst!=UINT32_MAX)
                {  
                        // m_weight_array[edge] = (meta_data.m_src + dst) % 128;
                        m_app_inst.AccumulateBuffer_del(meta_data.m_src,
                                                         dst,
                                                        &m_parent_array[dst],
                                                        &m_buffer_array[dst],
                                                        &m_value_array[dst]);
                         //insert edges to buffer
                }

                return true;
            }

        };
    } // namespace kernel
} // namespace sepgraph
#endif //HYBRID_PUSH_FUNCTOR_H
