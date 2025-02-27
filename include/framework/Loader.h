// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details


#include <algorithm>
#include <cinttypes>
#include <random>

#include "edgepair.h"
#include "pvector.h"
#include "groute/graphs/util.h"

// static const int64_t kRandSeed = 27491095;

template <typename NodeID_, typename DestID_ = NodeID_,
          typename WeightT_ = NodeID_>
class Loader {
  typedef EdgePair<NodeID_, DestID_> Edge;
  typedef WEdgePair<NodeID_, DestID_, WeightT_> WEdge;
  typedef Record<NodeID_, WeightT_> Record;
  typedef pvector<Edge> EdgeList;
  typedef pvector<WEdge> WEdgeList;
  using Hybrid = std::pair<uint32_t,uint32_t>;
 public:
 Loader(){}

  Loader(const std::string &filename, bool weight) {
    std::ifstream infile(filename);
    m_batch_size.reserve(10);
    uint32_t pos_batch = 0;
    std::stringstream ss;
    std::string line;
    uint32_t total_add = 0;
    uint32_t total_del = 0;
    while (getline(infile, line))
    {
        uint32_t add_size;
        uint32_t del_size;
        ss.str("");
        ss.clear();
        ss << line;
        ss >> add_size;
        ss >> del_size;
        m_batch_size[pos_batch].first = add_size;
        m_batch_size[pos_batch].second = del_size;
        pos_batch += 1;
        total_add += add_size;
        total_del += del_size;
    }
    infile.close();
    m_weight = weight;
    if(m_weight){
      added_edges_w.reserve(total_add);
      deleted_edges_w.reserve(total_del);
    }else{
      added_edges_.reserve(total_add);
      deleted_edges_.reserve(total_del);
    }
    m_add_size = total_add;
    m_del_size = total_del;

  }

  void ReadWeightList(const std::string &filename){
    std::ifstream infile(filename);
    std::stringstream ss;
    std::string line;
    uint32_t pos_add = 0;
    uint32_t pos_del = 0;
    // printf("ReadWeightList 2\n");
    while (getline(infile, line))
    {
        std::string type;
        uint32_t src,dst,weight;
        ss.str("");
        ss.clear();
        ss << line;
        ss >> type;
        ss >> src;
        ss >> dst;
        ss >> weight;
        // printf(" all src %d -> dst %d\n",src,dst);
        if(type == "a"){
          // printf(" add src %d -> dst %d\n",src,dst);
          // re[pos] = src;
          added_edges_w[pos_add].u = src;
          added_edges_w[pos_add].v = dst;
          added_edges_w[pos_add].w = weight;
          pos_add += 1;
        }else if(type == "d"){
          deleted_edges_w[pos_del].u = src;
          deleted_edges_w[pos_del].v = dst;
          deleted_edges_w[pos_del].w = weight;
          // printf(" del src %d -> dst %d  w %d\n",deleted_edges_w[pos_del].u,deleted_edges_w[pos_del].v,deleted_edges_w[pos_del].w);
          pos_del += 1;
        }

    }
    infile.close();
  }

  void ReadList(const std::string &filename){
    std::ifstream infile;
    // auto file = (char *) graphfile.c_str();
    infile.open(filename);
    std::stringstream ss;
    std::string line;
    uint32_t pos_add = 0;
    uint32_t pos_del = 0;
    while (getline(infile, line))
    {
        ss.str("");
        ss.clear();
        std::string type;
        uint32_t src,dst;
        ss << line;
        ss >> type;
        ss >> src;
        ss >> dst;
        if(type == "a"){
          added_edges_[pos_add].u = src;
          added_edges_[pos_add].v = dst;
          pos_add += 1;
        }else if(type == "d"){
          deleted_edges_[pos_del].u = src;
          deleted_edges_[pos_del].v = dst;
          pos_del += 1;
        }
    }
    infile.close();
  }

//  private:
public:
  uint32_t m_add_size;
  uint32_t m_del_size;
  std::vector<Hybrid> m_batch_size;
  bool m_weight;
  std::vector<WEdge> added_edges_w;
  std::vector<WEdge> deleted_edges_w;

  std::vector<Edge> added_edges_;
  std::vector<Edge> deleted_edges_;
};
