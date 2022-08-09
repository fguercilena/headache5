#ifndef HEADACHE5_HPP
#define HEADACHE5_HPP

#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#define H5_USE_16_API
#include "hdf5.h"

#define H5_THROW(M) throw std::runtime_error(M);

#define H5_CHECK_AND_THROW(S, M)                                               \
    if ((S) < 0)                                                               \
    {                                                                          \
        H5_THROW(M);                                                           \
    }

namespace H5
{
typedef std::vector<hsize_t> dimensions_t;

template <class T>
hid_t select_type()
{
    hid_t type;

    if (typeid(T) == typeid(float) or typeid(T) == typeid(double) or
        typeid(T) == typeid(long double))
    {
        type = H5T_NATIVE_DOUBLE;
    }
    else if (typeid(T) == typeid(short) or typeid(T) == typeid(int) or
             typeid(T) == typeid(long))
    {
        type = H5T_NATIVE_INT;
    }
    else if (typeid(T) == typeid(char))
    {
        type = H5T_NATIVE_CHAR;
    }
    else if (typeid(T) == typeid(bool))
    {
        type = H5T_NATIVE_HBOOL;
    }
    else
    {
        H5_THROW("Unsupported data type.");
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
        H5Sclose(m_dataspace_id);
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
        H5_CHECK_AND_THROW(m_dataspace_id,
                           "Failed to construct HDF5 dataspace.");
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
        using namespace std::string_literals;

        const htri_t status = H5Aexists(m_attributable_id, name.c_str());
        H5_CHECK_AND_THROW(status, "Problem reading attribute with name '"s +
                                       name + "'."s);

        return (status > 0);
    }

    template <class T>
    void write_attribute(const std::string& name, const T* data,
                         const dimensions_t& dimensions) const
    {
        using namespace std::string_literals;

        if (attribute_exists(name))
        {
            H5_THROW("Attribute with name '"s + name + "' already exists.");
        }

        DataSpace space(dimensions);

        const hid_t type = select_type<T>();

        // HDF5 compatibility
        const hid_t attribute_id = H5Acreate(m_attributable_id, name.c_str(),
                                             type, space.id(), H5P_DEFAULT);
        H5_CHECK_AND_THROW(attribute_id,
                           "Failed to create HDF5 attribute with name '"s +
                               name + "'."s);

        H5_CHECK_AND_THROW(H5Awrite(attribute_id, type, data),
                           "Failed to write to "
                           "HDF5 attribute with name '"s +
                               name + "'."s);

        H5_CHECK_AND_THROW(H5Aclose(attribute_id),
                           "Failed to close HDF5 attribute "
                           "with name '"s +
                               name + "'."s);
    }

    template <class T>
    std::pair<std::unique_ptr<T[]>, dimensions_t>
    read_attribute(const std::string& name) const
    {
        using namespace std::string_literals;

        if (not attribute_exists(name))
        {
            H5_THROW("Attribute with name '"s + name + "' does not exists.");
        }

        const hid_t attribute_id =
            H5Aopen(m_attributable_id, name.c_str(), H5P_DEFAULT);
        H5_CHECK_AND_THROW(attribute_id,
                           "Failed to open HDF5 attribute with name '"s + name +
                               "'."s);

        DataSpace space(H5Aget_space(attribute_id));

        const hid_t type = select_type<T>();

        void* buffer =
            static_cast<void*>(new char[space.size() * H5Tget_size(type)]);

        H5_CHECK_AND_THROW(H5Aread(attribute_id, type, buffer),
                           "Failed to read HDF5 "
                           "attribute with name '"s +
                               name + "'."s);

        H5_CHECK_AND_THROW(H5Aclose(attribute_id),
                           "Failed to close HDF5 attribute "
                           "with name '"s +
                               name + "'."s);

        return {std::unique_ptr<T[]>(static_cast<T*>(buffer)),
                space.dimensions()};
    }
};

template <class T>
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
        H5Dflush(m_dataset_id);
        H5Dclose(m_dataset_id);
    }

    DataSet(const hid_t group_id, const std::string& name,
            const DataSpace& space, dimensions_t chunks = {})
        : m_dimensions(space.dimensions()), m_name(name)
    {
        using namespace std::string_literals;

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
            H5_THROW("Invalid chunk size (does not correspond to the "
                     "dataset size).");
        }

        m_type = select_type<T>();

        hid_t plist_id;
        if (compress or m_extendable)
        {
            plist_id      = H5Pcreate(H5P_DATASET_CREATE);
            herr_t status = H5Pset_chunk(plist_id, rank, m_chunks.data());
            status        = H5Pset_deflate(plist_id, 9);

            H5_CHECK_AND_THROW(status, "Couldn't set chunk size and/or "
                                       "compression");
        }
        else
        {
            plist_id = H5P_DEFAULT;
        }

        // HDF5 compatibility
        m_dataset_id =
            H5Dcreate(group_id, name.c_str(), m_type, space.id(), plist_id);
        H5_CHECK_AND_THROW(m_dataset_id,
                           "Failed to create HDF5 dataset with name '"s + name +
                               "'."s);

        m_attributable_id = m_dataset_id;
    }

    DataSet(hid_t group_id, const std::string& name) : m_name(name)
    {
        using namespace std::string_literals;

        // HDF5 compatibility
        m_dataset_id = H5Dopen(group_id, name.c_str());
        H5_CHECK_AND_THROW(m_dataset_id,
                           "Failed to open HDF5 dataset with name '"s + name +
                               "'."s);

        m_type = select_type<T>();

        if (not H5Tequal(m_type, H5Dget_type(m_dataset_id)))
        {
            H5_THROW("Cannot construct H5::DataSet: H5::DataSet type and HDF5 "
                     "dataset type differ.");
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
        using namespace std::string_literals;

        if (not m_extendable)
        {
            H5_THROW("Data set '"s + m_name + "' is cannot be resized."s);
        }

        H5_CHECK_AND_THROW(H5Dextend(m_dataset_id, sizes.data()),
                           "Could not resize data set '"s + m_name + "'."s);

        m_dimensions = sizes;
    }

    template <class DT>
    void write_data(const DT* data) const
    {
        using namespace std::string_literals;

        const hid_t type = select_type<DT>();
        if (not H5Tequal(type, m_type))
        {
            H5_THROW("Tried to write data pointer with data type different "
                     "from dataset type.");
        }

        const herr_t status =
            H5Dwrite(m_dataset_id, m_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
        H5_CHECK_AND_THROW(status,
                           "Failed to write to HDF5 dataset with name '"s +
                               m_name + "'."s);
    }

    template <class DT>
    std::unique_ptr<DT[]>
    read_data(const dimensions_t& offset, const dimensions_t& stride,
              const dimensions_t& count, const dimensions_t& block) const
    {
        using namespace std::string_literals;

        if (offset.size() != rank() or stride.size() != rank() or
            count.size() != rank() or block.size() != rank())
        {
            H5_THROW("Wrong size for hyperslab parameters");
        }

        const hid_t type = select_type<DT>();
        if (not H5Tequal(type, m_type))
        {
            H5_THROW("Tried to read data with data type different from "
                     "dataset type.");
        }

        const hid_t fspace_id = H5Dget_space(m_dataset_id);
        H5_CHECK_AND_THROW(H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET,
                                               offset.data(), stride.data(),
                                               count.data(), block.data()),
                           "Couldn't select HDF5 hyperslab.");

        hid_t mspace_id    = H5Screate_simple(rank(), block.data(), nullptr);
        const hsize_t size = std::accumulate(block.begin(), block.end(), 1,
                                             std::multiplies<hsize_t>());
        void* buffer = static_cast<void*>(new char[size * H5Tget_size(m_type)]);

        herr_t status = H5Dread(m_dataset_id, m_type, mspace_id, fspace_id,
                                H5P_DEFAULT, buffer);
        H5_CHECK_AND_THROW(status, "Failed to read HDF5 dataset with name '"s +
                                       m_name + "'."s);

        return std::unique_ptr<DT[]>(static_cast<DT*>(buffer));
    }

    template <class DT>
    std::unique_ptr<DT[]> read_data(const dimensions_t& offset,
                                    const dimensions_t& block) const
    {
        const dimensions_t stride(m_dimensions.size(), 1);
        const dimensions_t count(m_dimensions.size(), 1);


        return read_data<DT>(offset, stride, count, block);
    }

    template <class DT>
    std::unique_ptr<DT[]> read_data() const
    {
        const dimensions_t offset(m_dimensions.size(), 0);

        return read_data<DT>(offset, m_dimensions);
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
        H5Gclose(m_group_id);
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
        using namespace std::string_literals;

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
        H5_CHECK_AND_THROW(new_group_id,
                           "Failed to create/open HDF5 group with "
                           "name '"s +
                               name + "'."s);

        return Group(new_group_id, name);
    }

    bool link_exists(const std::string& name) const
    {
        using namespace std::string_literals;

        htri_t status = H5Lexists(m_group_id, name.c_str(), H5P_DEFAULT);
        H5_CHECK_AND_THROW(status,
                           "Problem reading group/dataset with name '"s + name +
                               "'."s);

        return status > 0;
    }

    template <class T>
    DataSet<T> create_dataset(const std::string& name, const DataSpace& space,
                              dimensions_t chunks = {}) const
    {
        return DataSet<T>(m_group_id, name, space, chunks);
    }

    template <class T>
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
        H5Fflush(m_file_id, H5F_SCOPE_LOCAL);
        H5Fclose(m_file_id);
    }

    File(hid_t file_id) : Group(-1, "/"), m_file_id(file_id)
    {
        using namespace std::string_literals;

        m_attributable_id = m_file_id;

        // HDF5 compatibility
        m_group_id = H5Gopen(m_file_id, "/");
        H5_CHECK_AND_THROW(m_group_id, "Failed to open base group of file '"s +
                                           m_name + "'.");

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
        using namespace std::string_literals;

        const bool exists          = std::filesystem::is_regular_file(file);
        const htri_t is_accessible = exists ? H5Fis_hdf5(file.c_str()) : 0;

        if (mode == "r")
        {
            if (not exists or is_accessible <= 0)
            {
                H5_THROW("File '"s + file +
                         "' does not exists or cannot be accessed.");
            }
            m_file_id = H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        }
        if (mode == "r+")
        {
            if (not exists or is_accessible <= 0)
            {
                H5_THROW("File '"s + file +
                         "' does not exists or cannot be accessed.");
            }
            m_file_id = H5Fopen(file.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        }
        if (mode == "w")
        {
            m_file_id = H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                                  H5P_DEFAULT);
            H5_CHECK_AND_THROW(
                m_file_id, "Failed to create/overwrite HDF5 file with name '"s +
                               file + "'."s);
        }
        if (mode == "w-" or mode == "x")
        {
            if (exists)
            {
                H5_THROW("File '"s + file + "' already exists.");
            }
            m_file_id = H5Fcreate(file.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                                  H5P_DEFAULT);
        }
        if (mode == "a")
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
                H5_THROW("File '"s + file +
                         "' exists, but cannot be accessed.");
            }
        }

        m_attributable_id = m_file_id;

        // HDF5 compatibility
        m_group_id = H5Gopen(m_file_id, "/");
        H5_CHECK_AND_THROW(m_group_id, "Failed to open base group of file '"s +
                                           m_name + "'.");

        m_name = file;
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
