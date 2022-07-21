//************************************************************************
//  ExaMiniMD v. 1.0
//  Copyright (2018) National Technology & Engineering Solutions of Sandia,
//  LLC (NTESS).
//
//  Under the terms of Contract DE-NA-0003525 with NTESS, the U.S. Government
//  retains certain rights in this software.
//
//  ExaMiniMD is licensed under 3-clause BSD terms of use: Redistribution and
//  use in source and binary forms, with or without modification, are
//  permitted provided that the following conditions are met:
//
//    1. Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//
//    2. Redistributions in binary form must reproduce the above copyright notice,
//       this list of conditions and the following disclaimer in the documentation
//       and/or other materials provided with the distribution.
//
//    3. Neither the name of the Corporation nor the names of the contributors
//       may be used to endorse or promote products derived from this software
//       without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY EXPRESS OR IMPLIED
//  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//  IN NO EVENT SHALL NTESS OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
//  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
//  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
//  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//  Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//************************************************************************

// Runtime Check for this Neighbor Class
#ifdef MODULES_OPTION_CHECK
      if( (strcmp(argv[i+1], "CSR_MAPCONSTR") == 0) )
        neighbor_type = NEIGH_CSR_MAPCONSTR;
#endif

// Instantiation and Init of this class
#ifdef NEIGHBOR_MODULES_INSTANTIATION
    else if (input->neighbor_type == NEIGH_CSR_MAPCONSTR) {
      neighbor = new NeighborCSRMapConstr<t_neigh_mem_space>();
      neighbor->init(input->force_cutoff + input->neighbor_skin);
    }
#endif

// Add Force Instantiation case
#if defined(FORCE_MODULES_INSTANTIATION)
      case NEIGH_CSR_MAPCONSTR: force = new FORCETYPE_ALLOCATION_MACRO(NeighborCSRMapConstr<t_neigh_mem_space>); break;
#endif

// Add Force declaration line
#if defined(FORCE_MODULES_EXTERNAL_TEMPLATE)
      extern template class FORCETYPE_DECLARE_TEMPLATE_MACRO(NeighborCSRMapConstr<t_neigh_mem_space>);
#endif

// Add Force Template Instantiation line
#if defined(FORCE_MODULES_TEMPLATE)
      template class FORCETYPE_DECLARE_TEMPLATE_MACRO(NeighborCSRMapConstr<t_neigh_mem_space>);
#endif

// Making sure we are not just instantiating some Option
#if !defined(MODULES_OPTION_CHECK) && \
    !defined(NEIGHBOR_MODULES_INSTANTIATION) && \
    !defined(FORCE_MODULES_INSTANTIATION) && \
    !defined(FORCE_MODULES_EXTERNAL_TEMPLATE) && \
    !defined(FORCE_MODULES_TEMPLATE)
#ifndef NEIGHBOR_CSR_MAPCONSTR_H
#define NEIGHBOR_CSR_MAPCONSTR_H
#include <neighbor_csr.h>
#include <Kokkos_UnorderedMap.hpp>

template<class MemorySpace>
class NeighborCSRMapConstr: public Neighbor {

protected:
  T_X_FLOAT neigh_cut;
  t_x x;
  t_type type;
  t_id id;

  T_INT nbinx,nbiny,nbinz,nhalo;
  T_INT N_local;

  Kokkos::View<T_INT*, MemorySpace> num_neighs;
  Kokkos::View<T_INT*, MemorySpace, Kokkos::MemoryTraits<Kokkos::Atomic> > num_neighs_atomic;
  Kokkos::View<T_INT*, MemorySpace> neigh_offsets;
  Kokkos::View<T_INT*, MemorySpace> neighs;
  bool skip_num_neigh_count, half_neigh;

  typename Binning::t_binoffsets bin_offsets;
  typename Binning::t_bincount bin_count;
  typename Binning::t_permute_vector permute_vector;

  typedef Kokkos::UnorderedMap<Kokkos::pair<T_INT,T_INT>, void, MemorySpace> t_set;
  t_set pair_list;

public:
  struct TagFillPairListFull {};
  struct TagFillPairListHalf {};
  struct TagCopyNeighList {};
  struct TagCreateOffsets {};

  typedef Kokkos::TeamPolicy<TagFillPairListFull, Kokkos::IndexType<T_INT> > t_policy_fplf;
  typedef Kokkos::TeamPolicy<TagFillPairListHalf, Kokkos::IndexType<T_INT> > t_policy_fplh;
  typedef Kokkos::RangePolicy<TagCopyNeighList, Kokkos::IndexType<T_INT> > t_policy_cnl;
  typedef Kokkos::RangePolicy<TagCreateOffsets, Kokkos::IndexType<T_INT> > t_policy_co;

  typedef NeighListCSR<MemorySpace> t_neigh_list;

  t_neigh_list neigh_list;


  NeighborCSRMapConstr():neigh_cut(0.0) {
    neigh_type = NEIGH_CSR_MAPCONSTR;
  };
  ~NeighborCSRMapConstr() {};

  void init(T_X_FLOAT neigh_cut_) { neigh_cut = neigh_cut_; };

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagFillPairListFull&, const typename t_policy_fplf::member_type& team) const {
    const T_INT bx = team.league_rank()/(nbiny*nbinz) + nhalo;
    const T_INT by = (team.league_rank()/(nbinz)) % nbiny + nhalo;
    const T_INT bz = team.league_rank() % nbinz + nhalo;

    const T_INT i_offset = bin_offsets(bx,by,bz);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,bin_count(bx,by,bz)), [&] (const int bi) {
      const T_INT i = permute_vector(i_offset + bi);
      if(i>=N_local) return;
      const T_F_FLOAT x_i = x(i,0);
      const T_F_FLOAT y_i = x(i,1);
      const T_F_FLOAT z_i = x(i,2);
      const int type_i = type(i);

      for(int bx_j = bx-1; bx_j<bx+2; bx_j++)
      for(int by_j = by-1; by_j<by+2; by_j++)
      for(int bz_j = bz-1; bz_j<bz+2; bz_j++) {

        const T_INT j_offset = bin_offsets(bx_j,by_j,bz_j);
        int neigh_count_temp;
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, bin_count(bx_j,by_j,bz_j)), [&] (const T_INT bj, int& count) {
          T_INT j = permute_vector(j_offset + bj);
          const T_F_FLOAT dx = x_i - x(j,0);
          const T_F_FLOAT dy = y_i - x(j,1);
          const T_F_FLOAT dz = z_i - x(j,2);

          const int type_j = type(j);
          const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

          if((rsq <= neigh_cut*neigh_cut) && (i!=j)) {
            pair_list.insert(Kokkos::pair<T_INT,T_INT>(i,j));
            count++;
          }
        },neigh_count_temp);
        Kokkos::single(Kokkos::PerThread(team), [&] () {
          if(!skip_num_neigh_count)
            num_neighs_atomic(i)+=neigh_count_temp;
        });
      }
    });
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagFillPairListHalf&, const typename t_policy_fplh::member_type& team) const {
    const T_INT bx = team.league_rank()/(nbiny*nbinz) + nhalo;
    const T_INT by = (team.league_rank()/(nbinz)) % nbiny + nhalo;
    const T_INT bz = team.league_rank() % nbinz + nhalo;

    const T_INT i_offset = bin_offsets(bx,by,bz);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,0,bin_count(bx,by,bz)), [&] (const int bi) {
      const T_INT i = permute_vector(i_offset + bi);
      if(i>=N_local) return;
      const T_F_FLOAT x_i = x(i,0);
      const T_F_FLOAT y_i = x(i,1);
      const T_F_FLOAT z_i = x(i,2);
      const int type_i = type(i);

      for(int bx_j = bx-1; bx_j<bx+2; bx_j++)
      for(int by_j = by-1; by_j<by+2; by_j++)
      for(int bz_j = bz-1; bz_j<bz+2; bz_j++) {

        const T_INT j_offset = bin_offsets(bx_j,by_j,bz_j);
        int neigh_count_temp;
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, bin_count(bx_j,by_j,bz_j)), [&] (const T_INT bj, int& count) {
          T_INT j = permute_vector(j_offset + bj);
          const T_F_FLOAT x_j = x(j,0);
          const T_F_FLOAT y_j = x(j,1);
          const T_F_FLOAT z_j = x(j,2);
          if( ((j==i) || (j<N_local || comm_newton==true)) && !((x_j > x_i)  || ((x_j == x_i) && ( (y_j>y_i) ||  ((y_j==y_i) && (z_j>z_i) )))))
            return;
          const T_F_FLOAT dx = x_i - x_j;
          const T_F_FLOAT dy = y_i - y_j;
          const T_F_FLOAT dz = z_i - z_j;

          const int type_j = type(j);
          const T_F_FLOAT rsq = dx*dx + dy*dy + dz*dz;

          if((rsq <= neigh_cut*neigh_cut) && (i!=j)) {
            pair_list.insert(Kokkos::pair<T_INT,T_INT>(i,j));
            count++;
          }
        },neigh_count_temp);
        Kokkos::single(Kokkos::PerThread(team), [&] () {
          if(!skip_num_neigh_count)
            num_neighs_atomic(i)+=neigh_count_temp;
        });
      }
    });
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const TagCopyNeighList&, const T_INT& i) const {
    if(pair_list.valid_at(i)) {
      Kokkos::pair<T_INT,T_INT> pair_i = pair_list.key_at(i);
      T_INT offset = Kokkos::atomic_fetch_add(&num_neighs(pair_i.first),1) + neigh_offsets(pair_i.first);
      neighs(offset) = pair_i.second;
    }
  }


  KOKKOS_INLINE_FUNCTION
  void operator() (const TagCreateOffsets&, const T_INT& i, T_INT& offset, const bool final) const {
    const T_INT count_i = num_neighs(i);
    if(final)
      neigh_offsets(i) = offset;
    offset += count_i;
  }

  void create_neigh_list(System* system, Binning* binning, bool half_neigh_, bool) {
    // Get some data handles
    N_local = system->N_local;
    x = system->x;
    type = system->type;
    id = system->id;
    half_neigh = half_neigh_;

    T_INT total_num_neighs;

    // Compute number of expected neighbors
    T_X_FLOAT volume = system->domain_x * system->domain_y * system->domain_z;
    T_X_FLOAT particle_density =  1.0 * N_local / volume;
    T_INT num_neighs_estimate = particle_density *  (3.0/4.0 * 3.15 * neigh_cut * neigh_cut * neigh_cut);

    // Clear the PairList
    pair_list.clear();

    // Resize the PairList if necessary (make it a bit larger than needed for performance)
    if(pair_list.capacity() < T_INT(1.3 * num_neighs_estimate * N_local))
      pair_list.rehash( T_INT(1.3 * num_neighs_estimate * N_local) );

    // Reset the neighbor count array
    if( num_neighs.extent(0) < N_local + 1 ) {
      num_neighs = Kokkos::View<T_INT*, MemorySpace>("NeighborsCSRMapConstr::num_neighs", N_local + 1);
      neigh_offsets = Kokkos::View<T_INT*, MemorySpace>("NeighborsCSRMapConstr::neigh_offsets", N_local + 1);
    } else
      Kokkos::deep_copy(num_neighs,0);
    num_neighs_atomic = num_neighs;

    // Create the pair list
    nhalo = binning->nhalo;
    nbinx = binning->nbinx - 2*nhalo;
    nbiny = binning->nbiny - 2*nhalo;
    nbinz = binning->nbinz - 2*nhalo;

    T_INT nbins = nbinx*nbiny*nbinz;

    bin_offsets = binning->binoffsets;
    bin_count = binning->bincount;
    permute_vector = binning->permute_vector;

    skip_num_neigh_count = false;

    if(half_neigh)
      Kokkos::parallel_for("NeighborCSRMapConstr::fill_pair_list_half", t_policy_fplh(nbins,Kokkos::AUTO,8),*this);
    else
      Kokkos::parallel_for("NeighborCSRMapConstr::fill_pair_list_full", t_policy_fplf(nbins,Kokkos::AUTO,8),*this);
    Kokkos::fence();

    // Create the Offset list for neighbors of atoms
    Kokkos::parallel_scan("NeighborCSRMapConstr::create_offsets", t_policy_co(0, N_local+1), *this);
    Kokkos::fence();

    // Get the total neighbor count
    Kokkos::View<T_INT,MemorySpace> d_total_num_neighs(neigh_offsets,N_local);
    Kokkos::deep_copy(total_num_neighs,d_total_num_neighs);

    // If inserts failed because the PairList was to small resize and recreate the PairList
    // Don't need to count this time around
    if(pair_list.failed_insert()) {
      pair_list.rehash(total_num_neighs*1.2);
      skip_num_neigh_count = true;
      if(half_neigh)
        Kokkos::parallel_for("NeighborCSRMapConstr::fill_pair_list_half", t_policy_fplh(nbins,Kokkos::AUTO,8),*this);
      else
        Kokkos::parallel_for("NeighborCSRMapConstr::fill_pair_list_full", t_policy_fplf(nbins,Kokkos::AUTO,8),*this);
      Kokkos::fence();
    }

    // Resize NeighborList
    if( neighs.extent(0) < total_num_neighs )
      neighs = Kokkos::View<T_INT*, MemorySpace> ("NeighborCSRMapConstr::neighs", total_num_neighs);

    // Copy entries from the PairList to the actual NeighborList
    Kokkos::deep_copy(num_neighs,0);

    Kokkos::parallel_for("NeighborCSRMapConstr::copy_to_neigh_list",t_policy_cnl(0,pair_list.capacity()),*this);
    Kokkos::fence();

    // Create actual CSR NeighList
    neigh_list = t_neigh_list(
        Kokkos::View<T_INT*, MemorySpace>( neighs,     Kokkos::pair<T_INT,T_INT>(0,total_num_neighs)),
        Kokkos::View<T_INT*, MemorySpace>( neigh_offsets, Kokkos::pair<T_INT,T_INT>(0,N_local+1)));

  }

  t_neigh_list get_neigh_list() { return neigh_list; }

  const char* name() {return "NeighborCSRMapConstr";}
};

extern template struct NeighborCSRMapConstr<t_neigh_mem_space>;
#endif // #define NEIGHBOR_CSR_MAPCONSTR_H
#endif // MODULES_OPTION_CHECK / MODULES_INSTANTIATION
