#pragma once

#include <filesystem>
#include <format>
#include <numeric>
#include <print>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define H5_USE_16_API
#include "hdf5.h"


namespace H5
{

template <class E, class... Args>
inline void h5_throw(std::format_string<Args...> fmt, Args&&... args)
{
    std::string msg = std::vformat(fmt.get(), std::make_format_args(args...));

    throw E(msg);
}

template <class E, class F, class... Args>
inline void h5_check_and_throw(F flag, std::format_string<Args...> fmt,
                               Args&&... args)
{
    if (flag < 0)
    {
        h5_throw<E>(fmt, args...);
    }
}

template <class F, class... Args>
inline void h5_check_and_exit(F flag, std::format_string<Args...> fmt,
                              Args&&... args)
{
    if (flag < 0)
    {
        std::println(stderr, fmt, args...);
        std::fflush(stderr);
        std::exit(1);
    }
}

typedef std::vector<hsize_t> dimensions_t;

// Exceptions
/// @brief Exception raised for file-related HDF5 failures.
class FileError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

/// @brief Exception raised for group-related HDF5 failures.
class GroupError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

/// @brief Exception raised for dataset-related HDF5 failures.
class DataSetError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

/// @brief Exception raised for dataspace-related HDF5 failures.
class DataSpaceError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

/// @brief Exception raised for attribute-related HDF5 failures.
class AttributeError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

/// @brief Exception raised when an unsupported type is requested.
class TypeError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

/// @brief Exception raised for low-level I/O errors during HDF5 access.
class IOError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};


template <typename T>
inline constexpr hid_t select_HDF5_type()
{
    using namespace std;

    hid_t type = -1;

    if constexpr (is_same<T, float>::value)
    {
        type = H5T_NATIVE_FLOAT;
    }
    else if constexpr (is_same<T, double>::value or
                       is_same<T, long double>::value)
    {
        type = H5T_NATIVE_DOUBLE;
    }
    else if constexpr (is_same<T, short>::value or is_same<T, int>::value or
                       is_same<T, long>::value)
    {
        type = H5T_NATIVE_INT;
    }
    else if constexpr (is_same<T, char>::value)
    {
        type = H5T_NATIVE_CHAR;
    }
    else if constexpr (is_same<T, bool>::value)
    {
        type = H5T_NATIVE_HBOOL;
    }
    else
    {
        h5_throw<TypeError>("Type with typeid.name={:s} is not supported.",
                            typeid(T).name());
    }

    return type;
}

/// @brief Represents an HDF5 dataspace and its dimensional metadata.
///
/// A dataspace describes the shape of an HDF5 dataset or attribute and can
/// optionally be marked as extendable.
class DataSpace
{
  private:
    hid_t m_dataspace_id;
    dimensions_t m_dimensions;
    bool m_extendable;

  public:
    /// @brief Closes the underlying HDF5 dataspace if it is still valid.
    ~DataSpace()
    {
        if (H5Iis_valid(m_dataspace_id) > 0)
        {
#ifdef HEADACHE5_DEBUG
            std::println("Closing HDF5 dataspace with id '{:d}'.\n",
                         m_dataspace_id);
            std::fflush(stdout);
#endif
            h5_check_and_exit(H5Sclose(m_dataspace_id),
                              "Failed to close HDF5 dataspace with id '{:d}'.",
                              m_dataspace_id);
        }
    }

    /// @brief Constructs an invalid empty dataspace.
    DataSpace()
        : m_dataspace_id(H5I_INVALID_HID), m_dimensions({
                                               0,
                                           }),
          m_extendable(false)
    {
    }

    /// @brief Constructs a dataspace with the given dimensionality.
    /// @param d The dimensions of the dataspace.
    /// @param extendable Whether the dataspace may be resized later.
    DataSpace(const dimensions_t& d, const bool extendable = false)
        : m_dataspace_id(
              std::accumulate(d.begin(), d.end(), 1,
                              std::multiplies<hsize_t>()) == 1 ?
                  H5Screate(H5S_SCALAR) :
                  H5Screate_simple(
                      d.size(), d.data(),
                      extendable ?
                          dimensions_t(d.size(), H5S_UNLIMITED).data() :
                          nullptr)),
          m_dimensions(d), m_extendable(extendable)
    {
        h5_check_and_throw<DataSpaceError>(
            H5Iis_valid(m_dataspace_id), "Failed to construct HDF5 dataspace.");
    }

    /// @brief Wraps an existing HDF5 dataspace identifier.
    /// @param id The native HDF5 dataspace handle.
    DataSpace(const hid_t id) : m_dataspace_id(id)
    {
        m_dimensions.resize(H5Sget_simple_extent_ndims(m_dataspace_id));
        H5Sget_simple_extent_dims(m_dataspace_id, m_dimensions.data(), nullptr);
    }

    /// @brief Returns the underlying HDF5 dataspace identifier.
    /// @return The native HDF5 handle for the dataspace.
    hid_t id() const
    {
        return m_dataspace_id;
    }

    /// @brief Returns the number of dimensions in the dataspace.
    /// @return The dataspace rank.
    hsize_t rank() const
    {
        return m_dimensions.size();
    }

    /// @brief Returns the dimensional shape of the dataspace.
    /// @return A vector containing the extent of each dimension.
    dimensions_t dimensions() const
    {
        return m_dimensions;
    }

    /// @brief Returns the total number of elements represented by the dataspace.
    /// @return The product of all dimension lengths.
    hsize_t size() const
    {
        return std::accumulate(m_dimensions.begin(), m_dimensions.end(), 1,
                               std::multiplies<hsize_t>());
    }

    /// @brief Reports whether the dataspace can be resized.
    /// @return `true` if the dataspace is extendable, otherwise `false`.
    bool extendable() const
    {
        return m_extendable;
    }
};

/// @brief Common base class for HDF5 objects that support attributes.
///
/// This class provides attribute read/write helpers for datasets and groups.
class Attributable
{
  protected:
    hid_t m_attributable_id;

  public:
    /// @brief Checks whether an attribute with the specified name exists.
    /// @param name The attribute name.
    /// @return `true` if the attribute exists, otherwise `false`.
    /// @throws IOError If the HDF5 lookup fails.
    bool attribute_exists(const std::string& name) const
    {
        const htri_t status = H5Aexists(m_attributable_id, name.c_str());
        h5_check_and_throw<IOError>(
            status, "Problem reading attribute with name '{:s}'.", name);
        return (status > 0);
    }

    /// @brief Writes a raw array to a new HDF5 attribute.
    /// @tparam T The element type stored in the attribute.
    /// @param name The attribute name.
    /// @param data Pointer to the data buffer.
    /// @param dimensions The shape of the attribute data.
    /// @throws AttributeError If the attribute cannot be created or closed.
    /// @throws IOError If the write operation fails.
    template <typename T>
    void write_attribute(const std::string& name, const T* data,
                         const dimensions_t& dimensions) const
    {
        if (attribute_exists(name))
        {
            h5_throw<IOError>("Attribute with name '{:s}' already exists.",
                              name);
        }

        DataSpace space(dimensions);

        const hid_t type = select_HDF5_type<T>();

        // HDF5 compatibility
        const hid_t attribute_id = H5Acreate1(m_attributable_id, name.c_str(),
                                              type, space.id(), H5P_DEFAULT);
        h5_check_and_throw<AttributeError>(
            attribute_id, "Failed to create HDF5 attribute with name '{:s}'.",
            name);

        h5_check_and_throw<IOError>(
            H5Awrite(attribute_id, type, data),
            "Failed to write to HDF5 attribute with name '{:s}'.", name);

        h5_check_and_throw<AttributeError>(
            H5Aclose(attribute_id),
            "Failed to close HDF5 attribute with name '{:s}'.", name);
    }

    /// @brief Reads an attribute as a newly allocated array.
    /// @tparam T The element type expected from the attribute.
    /// @param name The attribute name.
    /// @return A pair containing the array buffer and the attribute dimensions.
    /// @throws AttributeError If the attribute is missing or cannot be opened.
    /// @throws IOError If the read operation fails.
    template <typename T>
    std::pair<std::unique_ptr<T[]>, dimensions_t>
    read_attribute(const std::string& name) const
    {
        if (not attribute_exists(name))
        {
            h5_throw<AttributeError>(
                "Attribute with name '{:s}' does not exists.", name);
        }

        const hid_t attribute_id =
            H5Aopen(m_attributable_id, name.c_str(), H5P_DEFAULT);
        h5_check_and_throw<AttributeError>(
            attribute_id, "Failed to open HDF5 attribute with name '{:s}'.",
            name);

        DataSpace space(H5Aget_space(attribute_id));

        const hid_t type = select_HDF5_type<T>();

        void* buffer =
            static_cast<void*>(new char[space.size() * H5Tget_size(type)]);

        h5_check_and_throw<IOError>(
            H5Aread(attribute_id, type, buffer),
            "Failed to read HDF5 attribute with name '{:s}'.", name);

        h5_check_and_throw<AttributeError>(
            H5Aclose(attribute_id),
            "Failed to close HDF5 attribute with name '{:s}'.", name);

        return {std::unique_ptr<T[]>(static_cast<T*>(buffer)),
                space.dimensions()};
    }
};

/// @brief Represents an HDF5 dataset associated with a native HDF5 type.
///
/// The dataset stores typed values and supports metadata, resizing, and
/// hyperslab reads/writes.
template <typename T>
class DataSet : public Attributable
{
  protected:
    hid_t m_dataset_id;
    dimensions_t m_dimensions;
    dimensions_t m_chunks;
    std::string m_dataset_name;
    hid_t m_type;
    bool m_extendable;

  public:
    /// @brief Closes the dataset and flushes pending data.
    ~DataSet()
    {
        if (H5Iis_valid(m_dataset_id) > 0)
        {
#ifdef HEADACHE5_DEBUG
            std::println("Closing HDF5 dataset '{:s}'.", m_dataset_name);
            std::fflush(stdout);
#endif
            h5_check_and_exit(H5Dflush(m_dataset_id),
                              "Failed to flush HDF5 dataset '{:s}'.",
                              m_dataset_name);
            h5_check_and_exit(H5Dclose(m_dataset_id),
                              "Failed to close HDF5 dataset '{:s}'.",
                              m_dataset_name);
        }
    }

    /// @brief Creates a new dataset inside an HDF5 group.
    /// @tparam T The dataset element type.
    /// @param group_id The parent group handle.
    /// @param name The dataset name.
    /// @param space The dataspace describing dataset dimensions.
    /// @param chunks Optional manual chunking layout.
    /// @throws DataSetError If the dataset cannot be created.
    DataSet(const hid_t group_id, const std::string& name,
            const DataSpace& space, dimensions_t chunks = {})
        : m_dimensions(space.dimensions()), m_dataset_name(name)
    {
        const hsize_t rank = space.rank();
        bool m_extendable  = space.extendable();
        bool compress      = true;

        if (space.size() * sizeof(T) < 4096)
        {
            m_chunks = space.dimensions();
            compress = false;
        }
        else if (chunks.size() == 0)
        {
            chunks = space.dimensions();

            for (hsize_t i = 0; i < rank; ++i)
            {
                while (chunks[i] > 50)
                {
                    chunks[i] /= 2;
                }
            }

            m_chunks = chunks;
        }
        else if (chunks.size() != space.rank())
        {
            h5_throw<DataSetError>("Invalid chunk size ({:d}), different from "
                                   "dataspace size ({:d}).",
                                   chunks.size(), space.rank());
        }

        m_type = select_HDF5_type<T>();

        hid_t plist_id;
        if (compress or m_extendable)
        {
            plist_id      = H5Pcreate(H5P_DATASET_CREATE);
            herr_t status = H5Pset_chunk(plist_id, rank, m_chunks.data());
            status        = H5Pset_deflate(plist_id, 9);

            h5_check_and_throw<DataSetError>(
                status, "Couldn't set chunk size and/or compression.");
        }
        else
        {
            plist_id = H5P_DEFAULT;
        }

        // HDF5 compatibility
        m_dataset_id =
            H5Dcreate1(group_id, name.c_str(), m_type, space.id(), plist_id);
        h5_check_and_throw<DataSetError>(
            m_dataset_id, "Failed to create HDF5 dataset with name '{:s}'.",
            name);

        m_attributable_id = m_dataset_id;
    }

    /// @brief Opens an existing dataset from a group.
    /// @tparam T The expected element type.
    /// @param group_id The parent group handle.
    /// @param name The dataset name.
    /// @throws DataSetError If the dataset cannot be opened.
    /// @throws TypeError If the expected type does not match the stored HDF5 type.
    DataSet(hid_t group_id, const std::string& name) : m_dataset_name(name)
    {
        // HDF5 compatibility
        m_dataset_id = H5Dopen1(group_id, name.c_str());
        h5_check_and_throw<DataSetError>(
            m_dataset_id, "Failed to open HDF5 dataset with name '{:s}'.",
            name);

        m_type = select_HDF5_type<T>();

        if (H5Tequal(m_type, H5Dget_type(m_dataset_id)) < 0)
        {
            h5_throw<TypeError>("Cannot open H5::DataSet '{:s}': "
                                "H5::DataSet type (code: {:d}) and "
                                "HDF5 dataset type (code: {:d}) differ.",
                                name, m_type, H5Dget_type(m_dataset_id));
        }

        DataSpace space(H5Dget_space(m_dataset_id));

        m_dimensions.resize(H5Sget_simple_extent_ndims(space.id()));
        H5Sget_simple_extent_dims(space.id(), m_dimensions.data(), nullptr);

        m_attributable_id = m_dataset_id;
    }

    /// @brief Move-assignment operator.
    /// @param other The dataset to move from.
    /// @return Reference to this dataset.
    DataSet& operator=(DataSet&& other)
    {
        m_dataset_id   = std::move(other.m_dataset_id);
        m_dataset_name = std::move(other.m_dataset_name);
        m_dimensions   = std::move(other.m_dimensions);
        m_chunks       = std::move(other.m_chunks);
        m_extendable   = std::move(other.m_extendable);
        m_type         = std::move(other.m_type);

        m_attributable_id = std::move(other.m_attributable_id);

        other.m_dataset_id = -1;

        return *this;
    }

    /// @brief Returns the dataset name.
    /// @return The dataset name as a string.
    std::string name() const
    {
        return m_dataset_name;
    }

    /// @brief Returns the rank of the dataset.
    /// @return The number of dimensions.
    hsize_t rank() const
    {
        return m_dimensions.size();
    }

    /// @brief Returns the dataset dimensions.
    /// @return A vector containing the extent along each axis.
    dimensions_t dimensions() const
    {
        return m_dimensions;
    }

    /// @brief Returns the total number of elements in the dataset.
    /// @return The product of all extents.
    hsize_t size() const
    {
        return std::accumulate(m_dimensions.begin(), m_dimensions.end(), 1,
                               std::multiplies<hsize_t>());
    }

    /// @brief Reports whether the dataset can be resized.
    /// @return `true` if the dataset is extendable, otherwise `false`.
    bool extendable() const
    {
        return m_extendable;
    }

    /// @brief Resizes an extendable dataset.
    /// @param sizes The new extents for each dimension.
    /// @throws DataSetError If the dataset is not extendable or the resize fails.
    void resize(const dimensions_t sizes)
    {
        if (not m_extendable)
        {
            h5_throw<DataSetError>("Dataset '{:s}' cannot be resized.",
                                   m_dataset_name);
        }

        h5_check_and_throw<DataSetError>(H5Dextend(m_dataset_id, sizes.data()),
                                         "Could not resize data set '{:s}'.",
                                         m_dataset_name);

        m_dimensions = sizes;
    }

    // TODO: write a hyperslab capable version of this function
    /// @brief Writes a full dataset from a raw buffer.
    /// @tparam DT The type of the provided data pointer.
    /// @param data Pointer to the data to write.
    /// @throws TypeError If the provided type does not match the dataset type.
    /// @throws IOError If the dataset write fails.
    template <typename DT>
    void write_data(const DT* data) const
    {
        const hid_t type = select_HDF5_type<DT>();

        if (H5Tequal(type, m_type) < 0)
        {
            h5_throw<TypeError>(
                "Tried to write data pointer with data type (code: {:d}) "
                "different from dataset type (code: {:d}).",
                type, m_type);
        }

        const herr_t status =
            H5Dwrite(m_dataset_id, m_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        h5_check_and_throw<IOError>(
            status, "Failed to write to HDF5 dataset with name '{:s}'.",
            m_dataset_name);
    }

    /// @brief Reads a hyperslab from the dataset using explicit offset/stride/count/block arguments.
    /// @tparam DT The type to read into.
    /// @param offset The offset of the hyperslab.
    /// @param stride The stride of the selected region.
    /// @param count The number of blocks to read in each dimension.
    /// @param block The block size for each dimension.
    /// @return A pointer to the read data buffer.
    /// @throws DataSetError If the hyperslab parameters are invalid.
    /// @throws TypeError If the requested type differs from the dataset type.
    /// @throws IOError If the HDF5 read fails.
    template <typename DT>
    std::unique_ptr<DT[]>
    read_data(const dimensions_t& offset, const dimensions_t& stride,
              const dimensions_t& count, const dimensions_t& block) const
    {
        if (offset.size() != rank() or stride.size() != rank() or
            count.size() != rank() or block.size() != rank())
        {
            h5_throw<DataSetError>("Wrong size for hyperslab parameters.");
        }

        const hid_t type = select_HDF5_type<DT>();
        if (H5Tequal(type, m_type) < 0)
        {
            h5_throw<TypeError>(
                "Tried to read data with data type (code: {:d}) different "
                "from dataset type (code: {:d}).",
                type, m_type);
        }

        const hid_t fspace_id = H5Dget_space(m_dataset_id);
        h5_check_and_throw<DataSetError>(
            H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, offset.data(),
                                stride.data(), count.data(), block.data()),
            "Couldn't select HDF5 hyperslab.");

        hid_t mspace_id    = H5Screate_simple(rank(), block.data(), nullptr);
        const hsize_t size = std::accumulate(block.begin(), block.end(), 1,
                                             std::multiplies<hsize_t>());

#ifdef HEADACHE5_DEBUG
        std::println(
            "Creating buffer to read dataset.\nThe number of elements in "
            "the buffer will be: {:d}\nThe size of the datatype is {:d}\n",
            size, H5Tget_size(m_type));
        std::fflush(stdout);
#endif

        void* buffer = static_cast<void*>(new char[size * H5Tget_size(m_type)]);

        herr_t status = H5Dread(m_dataset_id, m_type, mspace_id, fspace_id,
                                H5P_DEFAULT, buffer);
        h5_check_and_throw<IOError>(
            status, "Failed to read HDF5 dataset with name '{:s}'.",
            m_dataset_name);

        return std::unique_ptr<DT[]>(static_cast<DT*>(buffer));
    }

    /// @brief Reads a dataset slice using an offset and block size.
    /// @tparam DT The type to read into.
    /// @param offset The starting offset of the block.
    /// @param block The dimensions of the block to read.
    /// @return A pointer to the read buffer.
    template <typename DT>
    std::unique_ptr<DT[]> read_data(const dimensions_t& offset,
                                    const dimensions_t& block) const
    {
        const dimensions_t stride(m_dimensions.size(), 1);
        const dimensions_t count(m_dimensions.size(), 1);

        return read_data<DT>(offset, stride, count, block);
    }

    /// @brief Reads the entire dataset into a newly allocated buffer.
    /// @tparam DT The type to read into.
    /// @return A pointer to the dataset contents.
    /// @throws TypeError If the request type differs from the stored dataset type.
    /// @throws IOError If the HDF5 read fails.
    template <typename DT>
    std::unique_ptr<DT[]> read_data() const
    {
        const hid_t type = select_HDF5_type<DT>();

        if (H5Tequal(type, m_type) < 0)
        {
            h5_throw<TypeError>(
                "Tried to read data with data type (code: {:d}) different "
                "from dataset type (code: {:d}).",
                type, m_type);
        }

        void* buffer =
            static_cast<void*>(new char[size() * H5Tget_size(m_type)]);

        herr_t status = H5Dread(m_dataset_id, m_type, H5S_ALL, H5S_ALL,
                                H5P_DEFAULT, buffer);
        h5_check_and_throw<IOError>(
            status, "Failed to read HDF5 dataset with name '{:s}'.",
            m_dataset_name);

        return std::unique_ptr<DT[]>(static_cast<DT*>(buffer));
    }
};

/// @brief Represents an HDF5 group and exposes group/dataset creation helpers.
///
/// Groups may contain subgroups, datasets, and attributes.
class Group : public Attributable
{
  protected:
    hid_t m_group_id;
    std::string m_group_name;

  public:
    /// @brief Closes the underlying HDF5 group if it is still valid.
    ~Group()
    {
        if (H5Iis_valid(m_group_id) > 0)
        {
#ifdef HEADACHE5_DEBUG
            std::println("Closing HDF5 group '{:s}'.\n", m_group_name);
            std::fflush(stdout);
#endif
            h5_check_and_exit(H5Gflush(m_group_id),
                              "Failed to flush HDF5 group '{:s}'.",
                              m_group_name);
            h5_check_and_exit(H5Gclose(m_group_id),
                              "Failed to close HDF5 group '{:s}'.",
                              m_group_name);
        }
    }

    /// @brief Wraps an existing HDF5 group handle.
    /// @param group_id The native HDF5 group identifier.
    /// @param name The group name.
    Group(const hid_t group_id, const std::string& name)
        : m_group_id(group_id), m_group_name(name)
    {
        m_attributable_id = group_id;
    }

    /// @brief Move-assignment operator for groups.
    /// @param other The group to move from.
    /// @return Reference to this group.
    Group& operator=(Group&& other)
    {
        m_group_id   = std::move(other.m_group_id);
        m_group_name = std::move(other.m_group_name);

        m_attributable_id = std::move(other.m_attributable_id);

        other.m_group_id = -1;

        return *this;
    }

    /// @brief Returns the group name.
    /// @return The group path or name.
    std::string name() const
    {
        return m_group_name;
    }

    /// @brief Returns the native HDF5 group identifier.
    /// @return The underlying HDF5 handle.
    hid_t id() const
    {
        return m_group_id;
    }

    /// @brief Ensures a subgroup exists, creating it if necessary.
    /// @param name The subgroup name.
    /// @return A `Group` object for the target subgroup.
    /// @throws GroupError If the group cannot be created or opened.
    Group require_group(const std::string& name) const
    {
        hid_t new_group_id;

        if (link_exists(name))
        {
            // HDF5 compatibility
            new_group_id = H5Gopen1(m_group_id, name.c_str());
        }
        else
        {
            // HDF5 compatibility
            new_group_id = H5Gcreate1(m_group_id, name.c_str(), 0);
        }
        h5_check_and_throw<GroupError>(
            new_group_id, "Failed to create/open HDF5 group with '{:s}'.",
            name);

        return Group(new_group_id, name);
    }

    /// @brief Checks whether a group member or dataset with the given name exists.
    /// @param name The link name to test.
    /// @return `true` if the link exists, otherwise `false`.
    /// @throws GroupError If inspecting the link fails.
    bool link_exists(const std::string& name) const
    {
        htri_t status = H5Lexists(m_group_id, name.c_str(), H5P_DEFAULT);
        h5_check_and_throw<GroupError>(
            status, "Problem reading group/dataset with name '{:s}'.", name);

        return status > 0;
    }

    /// @brief Creates a dataset inside the group.
    /// @tparam T The dataset element type.
    /// @param name The dataset name.
    /// @param space The dataspace describing the dataset shape.
    /// @param chunks Optional chunking selection.
    /// @return The created dataset.
    template <typename T>
    DataSet<T> create_dataset(const std::string& name, const DataSpace& space,
                              dimensions_t chunks = {}) const
    {
        return DataSet<T>(m_group_id, name, space, chunks);
    }

    /// @brief Opens an existing dataset from the group.
    /// @tparam T The expected dataset element type.
    /// @param name The dataset name.
    /// @return The opened dataset.
    /// @throws DataSetError If the dataset cannot be opened.
    template <typename T>
    DataSet<T> open_dataset(const std::string& name) const
    {
        return DataSet<T>(m_group_id, name);
    }

    /// @brief Recursively visits linked group members using an HDF5 callback.
    /// @param callback The HDF5 iteration callback.
    /// @param callback_data User-supplied state passed to the callback.
    void visit(H5L_iterate_t callback, void* callback_data)
    {
        // HDF5 compatibility
        H5Lvisit(m_group_id, H5_INDEX_NAME, H5_ITER_INC, callback,
                 callback_data);
    }

    /// @brief Iterates over linked group members using the provided callback.
    /// @param callback The HDF5 iteration callback.
    /// @param callback_data User-supplied state passed to the callback.
    void iterate(H5L_iterate_t callback, void* callback_data)
    {
        H5Literate(m_group_id, H5_INDEX_NAME, H5_ITER_INC, nullptr, callback,
                   callback_data);
    }
};

/// @brief Represents an HDF5 file and provides access to its root group.
///
/// This class behaves like a group wrapper around the file’s root group and
/// manages file open/close semantics.
class File : public Group
{
  protected:
    hid_t m_file_id;
    std::string m_file_name;

  public:
    /// @brief Closes the file and its associated root group.
    ~File()
    {
        if (H5Iis_valid(m_file_id) > 0)
        {
#ifdef HEADACHE5_DEBUG
            std::println("Closing HDF5 file '{:s}'.\n", m_file_name);
            std::fflush(stdout);
#endif
            h5_check_and_exit(
                H5Gflush(m_group_id),
                "Failed to flush HDF5 base group for file '{:s}'.",
                m_file_name);
            h5_check_and_exit(
                H5Gclose(m_group_id),
                "Failed to close HDF5 base group for file '{:s}'.",
                m_file_name);

            h5_check_and_exit(H5Fflush(m_file_id, H5F_SCOPE_LOCAL),
                              "Failed to flush HDF5 file '{:s}'.", m_file_name);
            h5_check_and_exit(H5Fclose(m_file_id),
                              "Failed to close HDF5 file '{:s}'.", m_file_name);
        }
    }

    /// @brief Constructs an invalid default file handle.
    File()
        : Group(-1, "/"), m_file_id(-1),
          m_file_name("INVALID - DEFAULT CONSTRUCTED")
    {
    }

    /// @brief Wraps an existing HDF5 file identifier.
    /// @param file_id The native HDF5 file handle.
    File(hid_t file_id) : Group(-1, "/"), m_file_id(file_id)
    {
        m_attributable_id = m_file_id;

        // HDF5 compatibility
        m_group_id = H5Gopen1(m_file_id, "/");
        h5_check_and_throw<GroupError>(
            m_group_id, "Failed to open base group of file '{:s}'.",
            m_file_name);
        m_group_name = "/";

        ssize_t namesize = H5Fget_name(m_file_id, nullptr, 0);
        char* name       = new char[namesize + 1];
        H5Fget_name(m_file_id, name, namesize + 1);
        m_file_name.clear();
        m_file_name.insert(0, name);
        delete[] name;
    }

    /// @brief Opens or creates an HDF5 file with the requested mode.
    /// @param file The file path.
    /// @param mode The file access mode. Supported values are: "r", "r+",
    ///             "w", "w-"/"x", and "a".
    /// @throws FileError If the file is missing, inaccessible, or cannot be created/opened.
    File(const std::string& file, const std::string& mode = "r")
        : Group(-1, "/")
    {
        const bool exists          = std::filesystem::is_regular_file(file);
        const htri_t is_accessible = exists ? H5Fis_hdf5(file.c_str()) : 0;

        if (mode == "r") // Readonly, file must exist (default)
        {
            if (not exists)
            {
                h5_throw<FileError>("File '{:s}' does not exists.", file);
            }
            if (is_accessible <= 0)
            {
                h5_throw<FileError>("File '{:s}' is not accessible.", file);
            }
            m_file_id = H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        }
        else if (mode == "r+") // Read/write, file must exist
        {
            if (not exists)
            {
                h5_throw<FileError>("File '{:s}' does not exists.", file);
            }
            if (is_accessible <= 0)
            {
                h5_throw<FileError>("File '{:s}' is not accessible.", file);
            }
            m_file_id = H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        }
        else if (mode == "w") // Create file, truncate if exists
        {
            m_file_id = H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                                  H5P_DEFAULT);
            if (m_file_id <= 0)
            {
                h5_throw<FileError>(
                    "Failed to create/overwrite HDF5 file with name '{:s}'.",
                    file);
            }
        }
        else if (mode == "w-" or mode == "x") // Create file, fail if exists
        {
            if (exists)
            {
                h5_throw<FileError>("File '{:s}' already exists.", file);
            }
            m_file_id = H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                                  H5P_DEFAULT);
        }
        else if (mode == "a") // Read/write if exists, create otherwise
        {
            if (exists and is_accessible > 0)
            {
                m_file_id = H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
            }
            else if (not exists)
            {
                m_file_id = H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                                      H5P_DEFAULT);
            }
            else if (exists and is_accessible < 0)
            {
                h5_throw<FileError>(
                    "File '{:s}' exists, but cannot be accessed.", file);
            }
        }

        m_attributable_id = m_file_id;

        // HDF5 compatibility
        m_group_id = H5Gopen1(m_file_id, "/");
        h5_check_and_throw<GroupError>(
            m_group_id, "Failed to open base group of file '{:s}'.",
            m_file_name);
        m_group_name = "/";

        m_file_name = file;
    }

    /// @brief Move-assignment operator for files.
    /// @param other The file to move from.
    /// @return Reference to this file.
    File& operator=(File&& other)
    {
        m_file_id   = std::move(other.m_file_id);
        m_file_name = std::move(other.m_file_name);

        // m_group_id   = std::move(other.m_group_id);
        // m_group_name = std::move(other.m_group_name);
        Group::operator=(std::move(other));

        m_attributable_id = std::move(other.m_attributable_id);

        other.m_file_id = -1;

        return *this;
    }

    /// @brief Returns the file path.
    /// @return The file name or path as a string.
    std::string name() const
    {
        return m_file_name;
    }

    /// @brief Returns the native HDF5 file identifier.
    /// @return The underlying HDF5 handle for the file.
    hid_t id() const
    {
        return m_file_id;
    }
};

} // namespace H5
