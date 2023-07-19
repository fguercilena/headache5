#ifndef HEADACHE5_HPP
#define HEADACHE5_HPP

#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define H5_USE_16_API
#include "hdf5.h"


namespace H5
{

template <typename E>
void h5_throw(const char* format, ...)
{
    char msg[5000];

    va_list args;
    va_start(args, format);
    std::vsprintf(msg, format, args);
    va_end(args);

    throw E(msg);
}

template <typename E, typename F>
void h5_check_and_throw(F flag, const char* format, ...)
{
    if (flag < 0)
    {
        char msg[5000];

        va_list args;
        va_start(args, format);
        std::vsprintf(msg, format, args);
        va_end(args);

        throw E(msg);
    }
}

typedef std::vector<hsize_t> dimensions_t;

// Exceptions
class FileError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

class GroupError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

class DataSetError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

class DataSpaceError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

class AttributeError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

class TypeError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

class IOError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};


template <typename T>
constexpr hid_t select_HDF5_type()
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
        h5_throw<TypeError>("Type with typeid.name=%s is not supported.",
                            typeid(T).name());
    }

    return type;
}

class DataSpace
{
  private:
    hid_t m_dataspace_id;
    dimensions_t m_dimensions;
    bool m_extendable;

  public:
    ~DataSpace()
    {
        if (H5Iis_valid(m_dataspace_id) > 0)
        {
            H5Sclose(m_dataspace_id);
        }
    }

    DataSpace()
        : m_dataspace_id(H5I_INVALID_HID), m_dimensions({
                                               0,
                                           }),
          m_extendable(false)
    {
    }

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
            H5Iis_valid(m_dataspace_id), "Failed to construct HDF5 dataspace.",
            nullptr);
    }

    DataSpace(const hid_t id) : m_dataspace_id(id)
    {
        m_dimensions.resize(H5Sget_simple_extent_ndims(m_dataspace_id));
        H5Sget_simple_extent_dims(m_dataspace_id, m_dimensions.data(), nullptr);
    }

    hid_t id() const
    {
        return m_dataspace_id;
    }

    hsize_t rank() const
    {
        return m_dimensions.size();
    }

    dimensions_t dimensions() const
    {
        return m_dimensions;
    }

    hsize_t size() const
    {
        return std::accumulate(m_dimensions.begin(), m_dimensions.end(), 1,
                               std::multiplies<hsize_t>());
    }

    bool extendable() const
    {
        return m_extendable;
    }
};

class Attributable
{
  protected:
    hid_t m_attributable_id;

  public:
    bool attribute_exists(const std::string& name) const
    {
        const htri_t status = H5Aexists(m_attributable_id, name.c_str());
        h5_check_and_throw<IOError>(
            status, "Problem reading attribute with name '%s'.", name.c_str());
        return (status > 0);
    }

    template <typename T>
    void write_attribute(const std::string& name, const T* data,
                         const dimensions_t& dimensions) const
    {
        if (attribute_exists(name))
        {
            h5_throw<IOError>("Attribute with name '%s' already exists.",
                              name.c_str());
        }

        DataSpace space(dimensions);

        const hid_t type = select_HDF5_type<T>();

        // HDF5 compatibility
        const hid_t attribute_id = H5Acreate(m_attributable_id, name.c_str(),
                                             type, space.id(), H5P_DEFAULT);
        h5_check_and_throw<AttributeError>(
            attribute_id, "Failed to create HDF5 attribute with name '%s'.",
            name.c_str());

        h5_check_and_throw<IOError>(
            H5Awrite(attribute_id, type, data),
            "Failed to write to HDF5 attribute with name '%s'.", name.c_str());

        h5_check_and_throw<AttributeError>(
            H5Aclose(attribute_id),
            "Failed to close HDF5 attribute with name '%s'.", name.c_str());
    }

    template <typename T>
    std::pair<std::unique_ptr<T[]>, dimensions_t>
    read_attribute(const std::string& name) const
    {
        if (not attribute_exists(name))
        {
            h5_throw<AttributeError>(
                "Attribute with name '%s' does not exists.", name.c_str());
        }

        const hid_t attribute_id =
            H5Aopen(m_attributable_id, name.c_str(), H5P_DEFAULT);
        h5_check_and_throw<AttributeError>(
            attribute_id, "Failed to open HDF5 attribute with name '%s'.",
            name.c_str());

        DataSpace space(H5Aget_space(attribute_id));

        const hid_t type = select_HDF5_type<T>();

        void* buffer =
            static_cast<void*>(new char[space.size() * H5Tget_size(type)]);

        h5_check_and_throw<IOError>(
            H5Aread(attribute_id, type, buffer),
            "Failed to read HDF5 attribute with name '%s'.", name.c_str());

        h5_check_and_throw<AttributeError>(
            H5Aclose(attribute_id),
            "Failed to close HDF5 attribute with name '%s'.", name.c_str());

        return {std::unique_ptr<T[]>(static_cast<T*>(buffer)),
                space.dimensions()};
    }
};

template <typename T>
class DataSet : public Attributable
{
  protected:
    hid_t m_dataset_id;
    dimensions_t m_dimensions;
    dimensions_t m_chunks;
    std::string m_name;
    hid_t m_type;
    bool m_extendable;

  public:
    ~DataSet()
    {
        if (H5Iis_valid(m_dataset_id) > 0)
        {
            H5Dflush(m_dataset_id);
            H5Dclose(m_dataset_id);
        }
    }

    DataSet(const hid_t group_id, const std::string& name,
            const DataSpace& space, dimensions_t chunks = {})
        : m_dimensions(space.dimensions()), m_name(name)
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
            h5_throw<DataSetError>(
                "Invalid chunk size (%d), different from dataspace size (%d).",
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
                status, "Couldn't set chunk size and/or compression.", nullptr);
        }
        else
        {
            plist_id = H5P_DEFAULT;
        }

        // HDF5 compatibility
        m_dataset_id =
            H5Dcreate(group_id, name.c_str(), m_type, space.id(), plist_id);
        h5_check_and_throw<DataSetError>(
            m_dataset_id, "Failed to create HDF5 dataset with name '%s'.",
            name.c_str());

        m_attributable_id = m_dataset_id;
    }

    DataSet(hid_t group_id, const std::string& name) : m_name(name)
    {
        // HDF5 compatibility
        m_dataset_id = H5Dopen(group_id, name.c_str());
        h5_check_and_throw<DataSetError>(
            m_dataset_id, "Failed to open HDF5 dataset with name '%s'.",
            name.c_str());

        m_type = select_HDF5_type<T>();

        if (H5Tequal(m_type, H5Dget_type(m_dataset_id)) < 0)
        {
            h5_throw<TypeError>("Cannot open H5::DataSet '%s': "
                                "H5::DataSet type (code: %ld) and "
                                "HDF5 dataset type (code: %ld) differ.",
                                name.c_str(), m_type,
                                H5Dget_type(m_dataset_id));
        }

        DataSpace space(H5Dget_space(m_dataset_id));

        m_dimensions.resize(H5Sget_simple_extent_ndims(space.id()));
        H5Sget_simple_extent_dims(space.id(), m_dimensions.data(), nullptr);

        m_attributable_id = m_dataset_id;
    }

    std::string name() const
    {
        return m_name;
    }

    hsize_t rank() const
    {
        return m_dimensions.size();
    }

    dimensions_t dimensions() const
    {
        return m_dimensions;
    }

    hsize_t size() const
    {
        return std::accumulate(m_dimensions.begin(), m_dimensions.end(), 1,
                               std::multiplies<hsize_t>());
    }

    bool extendable() const
    {
        return m_extendable;
    }

    void resize(const dimensions_t sizes)
    {
        if (not m_extendable)
        {
            h5_throw<DataSetError>("Data set '%s' cannot be resized.",
                                   m_name.c_str());
        }

        h5_check_and_throw<DataSetError>(H5Dextend(m_dataset_id, sizes.data()),
                                         "Could not resize data set '%s'.",
                                         m_name.c_str());

        m_dimensions = sizes;
    }

    template <typename DT>
    void write_data(const DT* data) const
    {
        const hid_t type = select_HDF5_type<DT>();

        if (H5Tequal(type, m_type) < 0)
        {
            h5_throw<TypeError>(
                "Tried to write data pointer with data type (code: %ld) "
                "different from dataset type (code: %ld).",
                type, m_type);
        }

        const herr_t status =
            H5Dwrite(m_dataset_id, m_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        h5_check_and_throw<IOError>(
            status, "Failed to write to HDF5 dataset with name '%s'.",
            m_name.c_str());
    }

    template <typename DT>
    std::unique_ptr<DT[]>
    read_data(const dimensions_t& offset, const dimensions_t& stride,
              const dimensions_t& count, const dimensions_t& block) const
    {
        if (offset.size() != rank() or stride.size() != rank() or
            count.size() != rank() or block.size() != rank())
        {
            h5_throw<DataSetError>("Wrong size for hyperslab parameters.",
                                   nullptr);
        }

        const hid_t type = select_HDF5_type<DT>();
        if (H5Tequal(type, m_type) < 0)
        {
            h5_throw<TypeError>(
                "Tried to read data with data type (code: %ld) different "
                "from dataset type (code: %ld).",
                type, m_type);
        }

        const hid_t fspace_id = H5Dget_space(m_dataset_id);
        h5_check_and_throw<DataSetError>(
            H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, offset.data(),
                                stride.data(), count.data(), block.data()),
            "Couldn't select HDF5 hyperslab.", nullptr);

        hid_t mspace_id    = H5Screate_simple(rank(), block.data(), nullptr);
        const hsize_t size = std::accumulate(block.begin(), block.end(), 1,
                                             std::multiplies<hsize_t>());
        void* buffer = static_cast<void*>(new char[size * H5Tget_size(m_type)]);

        herr_t status = H5Dread(m_dataset_id, m_type, mspace_id, fspace_id,
                                H5P_DEFAULT, buffer);
        h5_check_and_throw<IOError>(
            status, "Failed to read HDF5 dataset with name '%s'.",
            m_name.c_str());

        return std::unique_ptr<DT[]>(static_cast<DT*>(buffer));
    }

    template <typename DT>
    std::unique_ptr<DT[]> read_data(const dimensions_t& offset,
                                    const dimensions_t& block) const
    {
        const dimensions_t stride(m_dimensions.size(), 1);
        const dimensions_t count(m_dimensions.size(), 1);


        return read_data<DT>(offset, stride, count, block);
    }

    template <typename DT>
    std::unique_ptr<DT[]> read_data() const
    {
        const hid_t type = select_HDF5_type<DT>();

        if (H5Tequal(type, m_type) < 0)
        {
            h5_throw<TypeError>(
                "Tried to read data with data type (code: %ld) different "
                "from dataset type (code: %ld).",
                type, m_type);
        }

        void* buffer =
            static_cast<void*>(new char[size() * H5Tget_size(m_type)]);

        herr_t status = H5Dread(m_dataset_id, m_type, H5S_ALL, H5S_ALL,
                                H5P_DEFAULT, buffer);
        h5_check_and_throw<IOError>(
            status, "Failed to read HDF5 dataset with name '%s'.",
            m_name.c_str());

        return std::unique_ptr<DT[]>(static_cast<DT*>(buffer));
    }
};

class Group : public Attributable
{
  protected:
    hid_t m_group_id;
    std::string m_name;

  public:
    ~Group()
    {
        if (H5Iis_valid(m_group_id) > 0)
        {
            H5Gclose(m_group_id);
        }
    }

    Group(const hid_t group_id, const std::string& name)
        : m_group_id(group_id), m_name(name)
    {
        m_attributable_id = group_id;
    }

    std::string name() const
    {
        return m_name;
    }

    hid_t id() const
    {
        return m_group_id;
    }

    Group require_group(const std::string& name) const
    {
        hid_t new_group_id;

        if (link_exists(name))
        {
            // HDF5 compatibility
            new_group_id = H5Gopen(m_group_id, name.c_str());
        }
        else
        {
            // HDF5 compatibility
            new_group_id = H5Gcreate(m_group_id, name.c_str(), 0);
        }
        h5_check_and_throw<GroupError>(
            new_group_id, "Failed to create/open HDF5 group with '%s'.",
            name.c_str());

        return Group(new_group_id, name);
    }

    bool link_exists(const std::string& name) const
    {
        htri_t status = H5Lexists(m_group_id, name.c_str(), H5P_DEFAULT);
        h5_check_and_throw<GroupError>(
            status, "Problem reading group/dataset with name '%s'.",
            name.c_str());

        return status > 0;
    }

    template <typename T>
    DataSet<T> create_dataset(const std::string& name, const DataSpace& space,
                              dimensions_t chunks = {}) const
    {
        return DataSet<T>(m_group_id, name, space, chunks);
    }

    template <typename T>
    DataSet<T> open_dataset(const std::string& name) const
    {
        return DataSet<T>(m_group_id, name);
    }

    void visit(H5L_iterate_t callback, void* callback_data)
    {
        // HDF5 compatibility
        H5Lvisit(m_group_id, H5_INDEX_NAME, H5_ITER_INC, callback,
                 callback_data);
    }

    void iterate(H5L_iterate_t callback, void* callback_data)
    {
        H5Literate(m_group_id, H5_INDEX_NAME, H5_ITER_INC, nullptr, callback,
                   callback_data);
    }
};

class File : public Group
{
  protected:
    hid_t m_file_id;
    std::string m_name;

  public:
    ~File()
    {
        if (H5Iis_valid(m_file_id) > 0)
        {
            H5Fflush(m_file_id, H5F_SCOPE_LOCAL);
            H5Fclose(m_file_id);
        }
    }

    File()
        : Group(-1, "/"), m_file_id(-1), m_name("INVALID - DEFAULT CONSTRUCTED")
    {
    }

    File(hid_t file_id) : Group(-1, "/"), m_file_id(file_id)
    {
        m_attributable_id = m_file_id;

        // HDF5 compatibility
        m_group_id = H5Gopen(m_file_id, "/");
        h5_check_and_throw<GroupError>(
            m_group_id, "Failed to open base group of file '%s'.",
            m_name.c_str());

        ssize_t namesize = H5Fget_name(m_file_id, nullptr, 0);
        char* name       = new char[namesize + 1];
        H5Fget_name(m_file_id, name, namesize + 1);
        m_name.clear();
        m_name.insert(0, name);
        delete[] name;
    }

    File(const std::string& file, const std::string& mode = "r")
        : Group(-1, "/")
    {
        const bool exists          = std::filesystem::is_regular_file(file);
        const htri_t is_accessible = exists ? H5Fis_hdf5(file.c_str()) : 0;

        if (mode == "r") // Readonly, file must exist (default)
        {
            if (not exists)
            {
                h5_throw<FileError>("File '%s' does not exists.", file.c_str());
            }
            if (is_accessible <= 0)
            {
                h5_throw<FileError>("File '%s' is not accessible.",
                                    file.c_str());
            }
            m_file_id = H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        }
        else if (mode == "r+") // Read/write, file must exist
        {
            if (not exists)
            {
                h5_throw<FileError>("File '%s' does not exists.", file.c_str());
            }
            if (is_accessible <= 0)
            {
                h5_throw<FileError>("File '%s' is not accessible.",
                                    file.c_str());
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
                    "Failed to create/overwrite HDF5 file with name '%s'.",
                    file.c_str());
            }
        }
        else if (mode == "w-" or mode == "x") // Create file, fail if exists
        {
            if (exists)
            {
                h5_throw<FileError>("File '%s' already exists.", file.c_str());
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
                h5_throw<FileError>("File '%s' exists, but cannot be accessed.",
                                    file.c_str());
            }
        }

        m_attributable_id = m_file_id;

        // HDF5 compatibility
        m_group_id = H5Gopen(m_file_id, "/");
        h5_check_and_throw<GroupError>(
            m_group_id, "Failed to open base group of file '%s'.",
            m_name.c_str());

        m_name = file;
    }

    File& operator=(const File& other)
    {
        m_file_id = other.m_file_id;
        m_name    = other.m_name;

        m_attributable_id = m_file_id;

        // HDF5 compatibility
        m_group_id = H5Gopen(m_file_id, "/");
        h5_check_and_throw<GroupError>(
            m_group_id, "Failed to open base group of file '%s'.",
            m_name.c_str());

        return *this;
    }

    std::string name() const
    {
        return m_name;
    }

    hid_t id() const
    {
        return m_file_id;
    }
};

} // namespace H5
#endif
