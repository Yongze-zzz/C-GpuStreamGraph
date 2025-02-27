// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details


#include <algorithm>
#include <cinttypes>
#include <random>

#include "edgepair.h"
#include "pvector.h"
#include "groute/graphs/util.h"


/*
GAP Benchmark Suite
Class:  Generator
Author: Scott Beamer

Given scale and degree, generates edgelist for synthetic graph
 - Intended to be called from Builder
 - GenerateEL(uniform) generates and returns the edgelist
 - Can generate uniform random (uniform=true) or R-MAT graph according
   to Graph500 parameters (uniform=false)
 - Can also randomize weights within a weighted edgelist (InsertWeights)
 - Blocking/reseeding is for parallelism with deterministic output edgelist
*/

// static const int64_t kRandSeed = 27491095;

template <typename NodeID_, typename DestID_ = NodeID_,
          typename WeightT_ = NodeID_>
class Generator {
  typedef EdgePair<NodeID_, DestID_> Edge;
  typedef WEdgePair<NodeID_, DestID_, WeightT_> WEdge;
  typedef pvector<Edge> EdgeList;
  typedef pvector<WEdge> WEdgeList;
 public:
  Generator(uint32_t scale, uint32_t degree) {
    scale_ = scale;
    num_nodes_ = 1l << scale;
    num_edges_ = num_nodes_ * degree;
    if (num_nodes_ > std::numeric_limits<NodeID_>::max()) {
      std::cout << "NodeID type (max: " << std::numeric_limits<NodeID_>::max();
      std::cout << ") too small to hold " << num_nodes_ << std::endl;
      std::cout << "Recommend changing NodeID (typedef'd in src/benchmark.h)";
      std::cout << " to a wider type and recompiling" << std::endl;
      std::exit(-31);
    }
    std::cout << " num_nodes_ = " <<num_nodes_<< std::endl;
    std::cout << " num_edges_ = " <<num_edges_<< std::endl;
  }

  void PermuteIDs(EdgeList &el) {
    pvector<NodeID_> permutation(num_nodes_);
    std::mt19937 rng(kRandSeed);
    #pragma omp parallel for
    for (NodeID_ n=0; n < num_nodes_; n++)
      permutation[n] = n;
    shuffle(permutation.begin(), permutation.end(), rng);
    #pragma omp parallel for
    for (uint64_t e=0; e < num_edges_; e++)
      el[e] = Edge(permutation[el[e].u], permutation[el[e].v]);
  }

  EdgeList MakeUniformEL() {
    EdgeList el(num_edges_);
    #pragma omp parallel
    {
      std::mt19937 rng;
      std::uniform_int_distribution<NodeID_> udist(0, num_nodes_-1);
      #pragma omp for
      for (uint64_t block=0; block < num_edges_; block+=block_size) {
        rng.seed(kRandSeed + block/block_size);
        for (uint64_t e=block; e < std::min(block+block_size, num_edges_); e++) {
          el[e] = Edge(udist(rng), udist(rng));
        }
      }
    }
    return el;
  }

  EdgeList MakeRMatEL() {
    const float A = 0.57f, B = 0.19f, C = 0.19f;
    EdgeList el(num_edges_);
    #pragma omp parallel
    {
      std::mt19937 rng;
      std::uniform_real_distribution<float> udist(0, 1.0f);
      #pragma omp for
      for (uint64_t block=0; block < num_edges_; block+=block_size) {
        rng.seed(kRandSeed + block/block_size);
        for (uint64_t e=block; e < std::min(block+block_size, num_edges_); e++) {
          NodeID_ src = 0, dst = 0;
          for (int depth=0; depth < scale_; depth++) {
            float rand_point = udist(rng);
            src = src << 1;
            dst = dst << 1;
            if (rand_point < A+B) {
              if (rand_point > A)
                dst++;
            } else {
              src++;
              if (rand_point > A+B+C)
                dst++;
            }
          }
          // printf("Rmat el src %d -> dst %d\n",src,dst);
          el[e] = Edge(src, dst);
        }
      }
    }
    PermuteIDs(el);
    // TIME_PRINT("Shuffle", std::shuffle(el.begin(), el.end(),
    //                                    std::mt19937()));
    return el;
  }

  EdgeList GenerateEL(bool uniform) {
    EdgeList el;
    // Timer t;
    // t.Start();
    if (uniform)
      el = MakeUniformEL();
    else
      el = MakeRMatEL();
    // t.Stop();
    // PrintTime("Generate Time", t.Seconds());
    return el;
  }

  // static void InsertWeights(pvector<EdgePair<NodeID_, NodeID_>> &el) {}

  // Overwrites existing weights with random from [1,255]
  WEdgeList InsertWeights(EdgeList &el) {
    WEdgeList wel(num_edges_);
    #pragma omp parallel
    {
      std::mt19937 rng;
      std::uniform_int_distribution<int> udist(1, 255);
      uint64_t el_size = el.size();
      #pragma omp for
      for (uint64_t block=0; block < el_size; block+=block_size) {
        rng.seed(kRandSeed + block/block_size);
        for (uint64_t e=block; e < std::min(block+block_size, el_size); e++) {
          // el[e].w = static_cast<WeightT_>(udist(rng)+1);    // to make sure weight is not zero
          wel[e].u = el[e].u;
          wel[e].v = el[e].v;
          wel[e].w = static_cast<WeightT_>(udist(rng)+1); 
          printf("wel src %d -> dst %d -> weight %d\n",wel[e].u,wel[e].v,wel[e].w);
        }
      }
    }
    return wel;
  }

//  private:
public:
  uint32_t scale_;
  uint64_t num_nodes_;
  uint64_t num_edges_;
  static const uint64_t block_size = 1<<18;
};
