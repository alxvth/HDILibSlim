/*
 *
 * Copyright (c) 2014, Nicola Pezzotti (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *  notice, this list of conditions and the following disclaimer in the
 *  documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *  must display the following acknowledgement:
 *  This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *  its contributors may be used to endorse or promote products derived from
 *  this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY NICOLA PEZZOTTI ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL NICOLA PEZZOTTI BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

#include "hd_joint_probability_generator_inl.h"

namespace hdi{
  namespace dr{
    template class HDJointProbabilityGenerator<float, std::vector<std::map<uint32_t, float>>>;
    // template class HDJointProbabilityGenerator<double,std::vector<std::map<uint32_t,double>>>;
    template class HDJointProbabilityGenerator<float, std::vector<std::map<uint32_t, float>>, int64_t>;
    template class HDJointProbabilityGenerator<float, std::vector<std::map<uint32_t, float>>, uint64_t>;

    template class HDJointProbabilityGenerator<float, std::vector<std::unordered_map<uint32_t, float>>>;
    // template class HDJointProbabilityGenerator<double,std::vector<std::unordered_map<uint32_t,double>>>;
    template class HDJointProbabilityGenerator<float, std::vector<std::unordered_map<uint32_t, float>>, int64_t>;
    template class HDJointProbabilityGenerator<float, std::vector<std::unordered_map<uint32_t, float>>, uint64_t>;

    template class HDJointProbabilityGenerator<float, std::vector<hdi::data::MapMemEff<uint32_t, float>>>;
    // template class HDJointProbabilityGenerator<double,std::vector<hdi::data::MapMemEff<uint32_t,double>>>;
    template class HDJointProbabilityGenerator<float, std::vector<hdi::data::MapMemEff<uint32_t, float>>, int64_t>;
    template class HDJointProbabilityGenerator<float, std::vector<hdi::data::MapMemEff<uint32_t, float>>, uint64_t>;

    template class HDJointProbabilityGenerator<float, std::vector<hdi::data::SparseVec<uint32_t, float>>>;
    // template class HDJointProbabilityGenerator<double,std::vector<hdi::data::SparseVec<uint32_t,double>>>;
    template class HDJointProbabilityGenerator<float, std::vector<hdi::data::SparseVec<uint32_t, float>>, int64_t>;
    template class HDJointProbabilityGenerator<float, std::vector<hdi::data::SparseVec<uint32_t, float>>, uint64_t>;
  }
}
