// include/device/DataStructureInfo.cuh
#ifndef DATA_STRUCTURE_INFO_CUH
#define DATA_STRUCTURE_INFO_CUH

#include <string>
#include <memory>

namespace cato {

/**
 * @brief Base class for data structure information used in kernel optimization
 * 
 * This abstract base class provides the interface for different types of
 * data structures (vectors, matrices, tensors) to communicate their
 * characteristics to the kernel launch optimizers.
 */
class DataStructureInfo {
public:
    virtual ~DataStructureInfo() = default;
    
    /**
     * @brief Get the dimensionality of the data structure
     * @return Number of dimensions (1 for vectors, 2 for matrices, etc.)
     */
    virtual int get_dimensionality() const = 0;
    
    /**
     * @brief Get the total number of elements
     * @return Total element count
     */
    virtual size_t get_total_elements() const = 0;
    
    /**
     * @brief Get the size of each element in bytes
     * @return Element size in bytes
     */
    virtual size_t get_element_size() const = 0;
    
    /**
     * @brief Get a string description of the structure type
     * @return Structure type name
     */
    virtual std::string get_structure_type() const = 0;
    
    /**
     * @brief Get the size of a specific dimension
     * @param dim Dimension index (0-based)
     * @return Size of the specified dimension
     */
    virtual size_t get_dimension_size(int dim) const = 0;
};

/**
 * @brief Information about 1D vector data structures
 */
class VectorInfo : public DataStructureInfo {
private:
    size_t size_;
    size_t element_size_;
    
public:
    VectorInfo(size_t size, size_t element_size) 
        : size_(size), element_size_(element_size) {}
    
    int get_dimensionality() const override { return 1; }
    size_t get_total_elements() const override { return size_; }
    size_t get_element_size() const override { return element_size_; }
    std::string get_structure_type() const override { return "Vector"; }
    size_t get_dimension_size(int dim) const override { 
        return (dim == 0) ? size_ : 0; 
    }
    
    size_t get_size() const { return size_; }
    size_t get_element_size_bytes() const { return element_size_; }
};

/**
 * @brief Information about 2D matrix data structures
 */
class MatrixInfo : public DataStructureInfo {
private:
    size_t rows_, cols_;
    size_t element_size_;
    
public:
    MatrixInfo(size_t rows, size_t cols, size_t element_size) 
        : rows_(rows), cols_(cols), element_size_(element_size) {}
    
    int get_dimensionality() const override { return 2; }
    size_t get_total_elements() const override { return rows_ * cols_; }
    size_t get_element_size() const override { return element_size_; }
    std::string get_structure_type() const override { return "Matrix"; }
    size_t get_dimension_size(int dim) const override { 
        return (dim == 0) ? rows_ : (dim == 1) ? cols_ : 0; 
    }
    
    size_t get_rows() const { return rows_; }
    size_t get_cols() const { return cols_; }
};

/**
 * @brief Information about 3D tensor data structures
 */
class Tensor3DInfo : public DataStructureInfo {
private:
    size_t dim0_, dim1_, dim2_;
    size_t element_size_;
    
public:
    Tensor3DInfo(size_t dim0, size_t dim1, size_t dim2, size_t element_size) 
        : dim0_(dim0), dim1_(dim1), dim2_(dim2), element_size_(element_size) {}
    
    int get_dimensionality() const override { return 3; }
    size_t get_total_elements() const override { return dim0_ * dim1_ * dim2_; }
    size_t get_element_size() const override { return element_size_; }
    std::string get_structure_type() const override { return "Tensor3D"; }
    size_t get_dimension_size(int dim) const override { 
        return (dim == 0) ? dim0_ : (dim == 1) ? dim1_ : (dim == 2) ? dim2_ : 0; 
    }
    
    size_t get_dim0() const { return dim0_; }
    size_t get_dim1() const { return dim1_; }
    size_t get_dim2() const { return dim2_; }
};

/**
 * @brief Information about N-dimensional tensor data structures
 */
class TensorNDInfo : public DataStructureInfo {
private:
    std::vector<size_t> dimensions_;
    size_t element_size_;
    
public:
    TensorNDInfo(const std::vector<size_t>& dimensions, size_t element_size) 
        : dimensions_(dimensions), element_size_(element_size) {}
    
    int get_dimensionality() const override { return dimensions_.size(); }
    size_t get_total_elements() const override { 
        size_t total = 1;
        for (size_t dim : dimensions_) total *= dim;
        return total;
    }
    size_t get_element_size() const override { return element_size_; }
    std::string get_structure_type() const override { return "TensorND"; }
    size_t get_dimension_size(int dim) const override { 
        return (dim >= 0 && dim < static_cast<int>(dimensions_.size())) ? dimensions_[dim] : 0; 
    }
    
    const std::vector<size_t>& get_dimensions() const { return dimensions_; }
};

} // namespace cato

#endif // DATA_STRUCTURE_INFO_CUH
