#include<stdio.h>
using namespace std;

template <typename SrcT, typename DstT = SrcT>
struct EdgePair {
  SrcT u;
  DstT v;

  EdgePair() {}

  EdgePair(SrcT u, DstT v) : u(u), v(v) {
    this->u = u;
    this->v = v;
  }
};
template <typename SrcT, typename DstT = SrcT, typename WeightT = SrcT>
struct WEdgePair {
  SrcT u;
  DstT v;
  WeightT w;
  WEdgePair() {}

  WEdgePair(SrcT u, DstT v, WeightT w) : u(u), v(v), w(w){
    this->u = u;
    this->v = v;
    this->w = w;

  }
};
template <typename SrcT, typename TimesT = SrcT>
struct Record {
  SrcT u;
  TimesT t;
  Record() {}

  Record(SrcT u, TimesT t) : u(u), t(t){
    this->u = u;
    this->t = t;
  }
};