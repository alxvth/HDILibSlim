#ifndef SPARSEMAT_H
#define SPARSEMAT_H

#include <Eigen/Sparse>


namespace hdi {
  namespace data {

    //DO NOT CHANGE THE INDICES OUT SIDE
    template <class Key, class T>
    class SparseVec {
    public:
      typedef Key         key_type;
      typedef T           mapped_type;
      typedef T           value_type;
      typedef Eigen::SparseVector<mapped_type> storage_type;

    public:
      //typedef typename storage_type::InnerIterator  iterator;
      //typedef typename storage_type::InnerIterator  const_iterator;

      void clear() { _memory.resize(0); }
      size_t size()const { return _memory.nonZeros(); }
      size_t capacity()const { return _memory.size(); }
      void shrink_to_fit() { _memory.data().squeeze(); }

      //iterators are always constant
      //iterator begin() { return iterator it(_memory); }
//      iterator end() { return _memory.end(); }
      //const_iterator begin()const { return iterator it(_memory); }
//      const_iterator end()const { return _memory.end(); }
      //const_iterator cbegin()const { return iterator it(_memory); }
//      const_iterator cend()const { return _memory.cend(); }

      //access
      mapped_type& operator[](const key_type& k) { return _memory.coeffRef(k); }

      auto coeff(const key_type& k) const { return _memory.coeff(k); }

      //!MEMORY ACCESS: With great power comes great responsibility!
      storage_type& memory() { return _memory; }
      //!MEMORY ACCESS: With great power comes great responsibility!
      const storage_type& memory()const { return _memory; }

    private:
      storage_type _memory;
    };

  }
}

#endif SPARSEMAT_H
