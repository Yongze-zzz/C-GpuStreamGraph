// ----------------------------------------------------------------
// SEP-Graph: Finding Shortest Execution Paths for Graph Processing under a Hybrid Framework on GPU
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE
// in the root directory of this source distribution.
// ----------------------------------------------------------------
#ifndef HYBRID_GRAPH_DATUM_H
#define HYBRID_GRAPH_DATUM_H

#include <gflags/gflags.h>
#include <framework/common.h>
#include <framework/hybrid_policy.h>
#include <groute/device/bitmap_impls.h>
#include <groute/graphs/csr_graph.cuh>
#include <groute/device/queue.cuh>
#include <utils/cuda_utils.h>
#include <vector>

#include <stdgpu/iterator.h>        // device_begin, device_end
#include <stdgpu/memory.h>          // createDeviceArray, destroyDeviceArray
#include <stdgpu/platform.h>        // STDGPU_HOST_DEVICE
#include <stdgpu/vector.cuh>
#include <stdgpu/unordered_map.cuh>
#include <cub/cub.cuh>
#define PRIORITY_SAMPLE_SIZE 1000

DECLARE_int32(SEGMENT);
DECLARE_double(wl_alloc_factor);
DECLARE_double(cache);
namespace sepgraph {
    namespace graphs {
        cub::CachingDeviceAllocator  g_allocator(true); 
        template<typename TValue,
                typename TBuffer,
                typename TWeight>
        struct GraphDatum {
            // Graph metadata
            uint32_t nnodes, nedges;
	               
            index_t segment = FLAGS_SEGMENT;
            index_t cache = FLAGS_cache;
            // Worklist
	        groute::Queue<index_t> m_wl_array_in_seg[512];
            groute::Queue<index_t> m_wl_array_in; // Work-list in
            groute::Queue<index_t> m_wl_array_out_high; // Work-list out High priority
            groute::Queue<index_t> m_wl_array_out_low; // Work-list out Low priority
            groute::Queue<index_t> m_wl_middle;
	    
	        //groute::Queue<index_t> m_wl_array_seg;
	        //std::vector< groute::Queue<index_t> > m_wl_array_total;
	        /*Code by AX range 39 to 41*/
	        std::vector<index_t> seg_active_num;
	        std::vector<index_t> seg_workload_num;
	        std::vector<TValue> seg_res_num;
            std::vector<index_t> seg_exc_list;
	    
            Bitmap m_wl_bitmap_in; // Work-list in
            Bitmap m_wl_bitmap_out_high; // Work-list out high
            Bitmap m_wl_bitmap_out_low; // Work-list out low
            Bitmap m_wl_bitmap_middle;

            utils::SharedValue<uint32_t> m_current_round;

            // In/Out-degree for every nodes
            utils::SharedArray<uint32_t> m_in_degree;
            utils::SharedArray<uint32_t> m_out_degree;

            // Total In/Out-degree
            utils::SharedValue<uint32_t> m_total_in_degree;
            utils::SharedValue<uint32_t> m_total_out_degree;
	        utils::SharedValue<uint32_t> m_seg_degree;
	        utils::SharedValue<TValue> m_seg_value;
            size_t temp_storage_bytes = 0;
            void *d_temp_storage      = NULL;   
            // Graph data
            // groute::graphs::single::NodeOutputDatum<TValue> m_node_value_datum;
            // groute::graphs::single::NodeOutputDatum<TBuffer> m_node_buffer_datum;
            // groute::graphs::single::NodeOutputDatum<uint32_t> m_node_parent_datum;
            TValue *m_node_value_datum;
            TBuffer *m_node_buffer_datum;
            TValue *m_node_parent_datum;
            TValue *m_node_level_datum;
            bool *m_node_reset_datum;
            std::vector<TValue> host_value;
            std::vector<TBuffer> host_buffer;
            std::vector<TValue> host_parent;
            // std::vector<bool> reset_node;
            groute::graphs::single::NodeOutputDatum<TBuffer> m_node_buffer_tmp_datum; // For sync algorithms
            groute::graphs::single::EdgeInputDatum<TWeight> m_csr_edge_weight_datum;
            groute::graphs::single::EdgeInputDatum<TWeight> m_csc_edge_weight_datum;
            groute::graphs::single::EdgeInputDatum<TWeight> m_vcsr_edge_weight_datum;

            //Subgraph data
            utils::SharedArray<uint32_t> activeNodesLabeling;
            utils::SharedArray<uint32_t> activeNodesDegree;
            utils::SharedArray<uint32_t> prefixLabeling;
            utils::SharedArray<uint32_t> prefixSumDegrees;
            uint32_t subgraphnodes,subgraphedges;
            uint32_t Compaction_num;
            // Running data
            utils::SharedValue<uint32_t> m_active_nodes;

            // Sampling
            utils::SharedArray<index_t> m_sampled_nodes;
            utils::SharedArray<TBuffer> m_sampled_values;
            bool m_weighted;
            bool m_on_pinned_memory;
            index_t priority_detal;
            index_t* d_v;
            index_t* d_sum;
            cub::DoubleBuffer<uint32_t> d_hotness;
            cub::DoubleBuffer<uint32_t>   d_id;
            index_t* host_cache;
            index_t* host_cache_l3;
            index_t* cache_edges_l1;
            index_t* cache_edges_l2;
            index_t* cache_edges_com;
            uint64_t num_of_cache;
            uint64_t* num_of_cache_d;
            stdgpu::unordered_map<index_t,bool> nodes_control;
            //============================================================
            uint64_t *count_gpu;
            uint64_t *count_cpu;
            uint64_t *total_act_d;
             uint64_t *total_act;
            bool* reset_node;
            //============================================================
            GraphDatum(const groute::graphs::host::PMAGraph &vcsr_graph,
		               uint64_t seg_max_edge,
                       std::vector<index_t> nnodes_num,
                                             bool OnPinnedMemory=true) : nnodes(vcsr_graph.nnodes),
                                                                          nedges(vcsr_graph.nedges),
                                                                          m_in_degree(nullptr, 0),
                                                                          m_out_degree(nullptr, 0),
                                                                          activeNodesDegree(nullptr, 0),
                                                                          activeNodesLabeling(nullptr, 0),
                                                                          prefixLabeling(nullptr, 0),
                                                                          prefixSumDegrees(nullptr, 0),
                                                                          m_sampled_nodes(nullptr, 0),
                                                                          m_sampled_values(nullptr, 0),
                                                                          m_on_pinned_memory(OnPinnedMemory){
                // m_node_value_datum.Allocate(vcsr_graph);
                // m_node_buffer_datum.Allocate(vcsr_graph);
                // m_node_buffer_tmp_datum.Allocate(vcsr_graph);
                GROUTE_CUDA_CHECK(cudaMalloc((void**)&m_node_value_datum, sizeof(TValue)*vcsr_graph.nnodes));
                GROUTE_CUDA_CHECK(cudaMalloc((void**)&m_node_buffer_datum, sizeof(TBuffer)*vcsr_graph.nnodes));
                GROUTE_CUDA_CHECK(cudaMalloc((void**)&m_node_parent_datum, sizeof(TValue)*vcsr_graph.nnodes));
                GROUTE_CUDA_CHECK(cudaMalloc((void**)&m_node_reset_datum, sizeof(bool)*vcsr_graph.nnodes));
                // GROUTE_CUDA_CHECK(cudaMalloc((void**)&m_node_tmp_buffer_datum, sizeof(TBuffer)*vcsr_graph.nnodes));
                // GROUTE_CUDA_CHECK(cudaMalloc((void**)&m_node_level_datum, sizeof(TValue)*vcsr_graph.nnodes));
                uint64_t unit_gb = 1073741824;
                if(cache == 4 ){
                    unit_gb = 1073741824;
                }else if(cache == 3){
                    unit_gb = 805306365;
                }else if(cache ==2){
                    unit_gb = 536870921;
                }else if(cache ==1){
                    unit_gb = 268435455;
                }else if(cache ==0){
                    unit_gb = 0;
                }
                num_of_cache = unit_gb;
                LOG("内存分配的边数量 %lu\n", num_of_cache);
                GROUTE_CUDA_CHECK(cudaMalloc((void**)&d_v, sizeof(index_t)*nnodes));//cache L1
                GROUTE_CUDA_CHECK(cudaMalloc((void**)&d_sum, sizeof(index_t)*nnodes));//cache L1
                GROUTE_CUDA_CHECK(cudaMalloc((void**)&cache_edges_l1, sizeof(index_t)*num_of_cache));//cache L1
                GROUTE_CUDA_CHECK(cudaMalloc((void**)&cache_edges_com, sizeof(index_t)*(num_of_cache)));// cache L3
                GROUTE_CUDA_CHECK(cudaMalloc((void**)&cache_edges_l2, sizeof(index_t)*(0)));// cache L2
                CubDebugExit(g_allocator.DeviceAllocate((void**)&d_hotness.d_buffers[0], sizeof(uint32_t) * vcsr_graph.nnodes));
                CubDebugExit(g_allocator.DeviceAllocate((void**)&d_hotness.d_buffers[1], sizeof(uint32_t) * vcsr_graph.nnodes));
                CubDebugExit(g_allocator.DeviceAllocate((void**)&d_id.d_buffers[0], sizeof(uint32_t) * vcsr_graph.nnodes));
                CubDebugExit(g_allocator.DeviceAllocate((void**)&d_id.d_buffers[1], sizeof(uint32_t) * vcsr_graph.nnodes));
                
                // this->count_cpu = (uint64_t*)malloc(sizeof(uint64_t)*nnodes);
                // memset(count_cpu, 0, sizeof(uint64_t)*nnodes);
                GROUTE_CUDA_CHECK(cudaMalloc((void**)&count_gpu, sizeof(uint64_t)*nnodes));//cache L1
                GROUTE_CUDA_CHECK(cudaMalloc((void**)&total_act_d, sizeof(uint64_t)*nnodes));//cache L1
                this->total_act = (uint64_t*)malloc(sizeof(uint64_t)*nnodes);
                this->reset_node = (bool*)malloc(sizeof(bool)*nnodes);
                memset(reset_node, false, sizeof(bool)*nnodes);
                memset(total_act, 0, sizeof(uint64_t)*nnodes);
                this->count_cpu = (uint64_t*)malloc(sizeof(uint64_t)*nnodes);
                memset(count_cpu, 0, sizeof(uint64_t)*nnodes);
                printf("缓存大小： %d\n",num_of_cache);
                GROUTE_CUDA_CHECK(cudaMalloc((void**)&num_of_cache_d, sizeof(uint64_t)*(1)));
                GROUTE_CUDA_CHECK(cudaMemcpy(num_of_cache_d,&num_of_cache, 1* sizeof(uint64_t),cudaMemcpyHostToDevice));
                host_cache = (index_t*)malloc(sizeof(index_t)*num_of_cache);
                host_cache_l3 = (index_t*)malloc(sizeof(index_t)*(num_of_cache));
                if (typeid(TWeight) != typeid(groute::graphs::NoWeight)) {
                    m_weighted = true;
                    LOG("Allocate node for GraphDatum weighted ? %d\n",m_weighted);
                }
                else{
		             m_weighted = false;
		        }   
                CubDebugExit(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_hotness, d_id, vcsr_graph.nnodes));
                CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
                uint32_t capacity = nnodes * FLAGS_wl_alloc_factor;

		        for(index_t i = 0; i < segment; i++){
		              m_wl_array_in_seg[i] = std::move(groute::Queue<index_t>(nnodes_num[i]));
		        }
                m_wl_array_in_seg[segment] = std::move(groute::Queue<index_t>(nnodes)); //for zero task combine
                m_wl_array_in_seg[segment + 1] = std::move(groute::Queue<index_t>(nnodes)); //for compaction task combine

		        seg_active_num = std::move(std::vector<index_t>(segment));
		        seg_workload_num = std::move(std::vector<index_t>(segment));
		        seg_res_num = std::move(std::vector<TValue>(segment));
		        seg_exc_list = std::move(std::vector<index_t>(segment));
                m_wl_bitmap_out_high = std::move(Bitmap(nnodes));
                // GROUTE_CUDA_CHECK(cudaMalloc(&activeNodesLabeling.dev_ptr, nnodes * sizeof(uint32_t)));
                // GROUTE_CUDA_CHECK(cudaMalloc(&activeNodesDegree.dev_ptr, nnodes * sizeof(uint32_t)));
                // GROUTE_CUDA_CHECK(cudaMalloc(&prefixLabeling.dev_ptr, nnodes * sizeof(uint32_t)));
                // GROUTE_CUDA_CHECK(cudaMalloc(&prefixSumDegrees.dev_ptr, (nnodes + 1) * sizeof(uint32_t)));

                // m_sampled_nodes = std::move(utils::SharedArray<index_t>(PRIORITY_SAMPLE_SIZE));
                // m_sampled_values = std::move(utils::SharedArray<TBuffer>(PRIORITY_SAMPLE_SIZE));
            }

            GraphDatum(GraphDatum &&other) = delete;

            GraphDatum &operator=(const GraphDatum &other) = delete;

            GraphDatum &operator=(GraphDatum &&other) = delete;

            // const groute::graphs::dev::GraphDatum<TValue &GetValueDeviceObject() const {
            TValue* GetValueDeviceObject()  {    
                return m_node_value_datum;
            }
            uint64_t* GetSizeDeviceObject()  {    
                return num_of_cache_d;
            }
            // const groute::graphs::dev::GraphDatum<uint32_t> &GetParentDeviceObject() const {
            TValue* GetParentDeviceObject() {        
                return m_node_parent_datum;
            }
            void CompareDeviceResult(){
                uint32_t *host_id = (uint32_t*) malloc(nnodes * sizeof(uint32_t));
                uint32_t *host_hot = (uint32_t*) malloc(nnodes * sizeof(uint32_t));
                 // Copy data back
                cudaMemcpy(host_id, d_id.Current(), sizeof(uint32_t) * nnodes, cudaMemcpyDeviceToHost);
                cudaMemcpy(host_hot, d_hotness.Current(), sizeof(uint32_t) * nnodes, cudaMemcpyDeviceToHost);
                for (uint32_t i = 0; i < 1000; i++)
                {
                    uint32_t h_id = host_id[i];
                    uint32_t hot = host_hot[i];
                    // std::cout << "v: "<<(host_id[i]) << ", hot: "<<(host_hot[i])<<std::endl;
                    printf("#%d# hotness %d\n",h_id,hot);
                }
                free(host_id);
                free(host_hot);
                
            }
            TValue* GetLevelDeviceObject() {        
                return m_node_level_datum;
            }
            void sort_vtx_by_hotness(){
            LOG("sort vtx by hot\n");
                // CubDebugExit(cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_hotness, d_id, nnodes));       
                (cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_hotness, d_id, nnodes));     
            }
            void ensure_candidate_vertex(){
                this->d_temp_storage = NULL;
                this->temp_storage_bytes = 0;
                CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_v, d_sum, nnodes)); 
                CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
                CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_v, d_sum, nnodes));
            }
            // const groute::graphs::dev::GraphDatum<TBuffer> &GetBufferDeviceObject() const {
            TBuffer* GetBufferDeviceObject(){     
                return m_node_buffer_datum;
            }
            // TBuffer* GetBufferTmpDeviceObject(){     
            //     return m_node_tmp_buffer_datum;
            // }
            // const groute::graphs::dev::GraphDatum<TBuffer> &GetBufferTmpDeviceObject() const {
            //     return m_node_buffer_tmp_datum.DeviceObject();
            // }

            const groute::graphs::dev::GraphDatum<TWeight> &GetEdgeWeightDeviceObject() const {
                // return m_csr_edge_weight_datum.DeviceObject();
                return m_vcsr_edge_weight_datum.DeviceObject();
            }

            const groute::graphs::dev::GraphDatum<TWeight> &GetCSCEdgeWeightDeviceObject() const {
                return m_csc_edge_weight_datum.DeviceObject();
            }

            const groute::dev::WorkSourceRange<index_t> GetWorkSourceRangeDeviceObject(index_t seg_snode, index_t seg_nnode) {
                return groute::dev::WorkSourceRange<index_t>(seg_snode, seg_nnode);
            }
	        const groute::dev::WorkSourceRange<index_t> GetWorkSourceRangeDeviceObject() {
                return groute::dev::WorkSourceRange<index_t>(0, nnodes);
            }

            void GatherValue() {
                host_value.resize(nnodes);
                GROUTE_CUDA_CHECK(cudaMemcpy(
                        &host_value[0], m_node_value_datum,
                        (nnodes) * sizeof(TValue), cudaMemcpyDeviceToHost));
            }

            void GatherCacheL1() {
                GROUTE_CUDA_CHECK(cudaMemcpy(
                        &host_cache[0], cache_edges_l1,
                        (num_of_cache) * sizeof(index_t), cudaMemcpyDeviceToHost));
            }

            void GatherCacheL3() {
                GROUTE_CUDA_CHECK(cudaMemcpy(
                        &host_cache_l3[0], cache_edges_com,
                        (num_of_cache) * sizeof(index_t), cudaMemcpyDeviceToHost));
            }   

            void GatherCacheMiss() {
                GROUTE_CUDA_CHECK(cudaMemcpy(this->count_cpu, this->count_gpu,(nnodes) * sizeof(uint64_t), cudaMemcpyDeviceToHost));

            }

            void Gathertransfer() {
                GROUTE_CUDA_CHECK(cudaMemcpy(this->total_act, this->total_act_d,(nnodes) * sizeof(uint64_t), cudaMemcpyDeviceToHost));

            }
            void Resettransfer(){
                GROUTE_CUDA_CHECK(cudaMemcpy(
                        this->total_act_d, this->total_act,
                        (nnodes) * sizeof(uint64_t), cudaMemcpyHostToDevice));
            }

            void ResetCacheMiss(){                    
                GROUTE_CUDA_CHECK(cudaMemcpy(
                        this->count_gpu, this->count_cpu,
                        (nnodes) * sizeof(uint64_t), cudaMemcpyHostToDevice));
            }

             void GatherParent() {
                host_parent.resize(nnodes);
                GROUTE_CUDA_CHECK(cudaMemcpy(
                        &host_parent[0], m_node_parent_datum,
                        (nnodes) * sizeof(TValue), cudaMemcpyDeviceToHost));
            }

            void GatherBuffer() {
                host_buffer.resize(nnodes);
                GROUTE_CUDA_CHECK(cudaMemcpy(
                        &host_buffer[0], m_node_buffer_datum,
                        (nnodes) * sizeof(TBuffer), cudaMemcpyDeviceToHost));
            }
            void GatherReset(){
                // reset_node.resize(nnodes);
                GROUTE_CUDA_CHECK(cudaMemcpy(
                        &reset_node[0], m_node_reset_datum,
                        (nnodes) * sizeof(bool), cudaMemcpyDeviceToHost));
            }

        };
    }
}

#endif //HYBRID_GRAPH_DATUM_H
