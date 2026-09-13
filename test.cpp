#include <cstring>
#include <print>
#include <random>
#include <vector>

#include "headache5.hpp"


using namespace std;
using namespace H5;


int main()
{
    print("Generating random data for testing... ");
    fflush(stdout);

    random_device rd;
    default_random_engine rng(rd());
    uniform_real_distribution<double> urd(-56., +268.9);

    vector<double> test_data;
    for (int i = 0; i < 16; ++i)
    {
        test_data.push_back(urd(rng));
    }

    const char test_attribute[] = "Greetings, human.";

    println("done.\n");
    fflush(stdout);


    ////////////////////////////////////////////////////////////////////////////

    try
    {
        print("Test opening a new file... ");
        fflush(stdout);
        File test_file("test_file.hdf5", "w");
        println("PASS.");
        fflush(stdout);

        print("Test creating a group... ");
        fflush(stdout);
        Group group = test_file.require_group("test_group");
        println("PASS.");
        fflush(stdout);

        print("Test creating a dataspace... ");
        fflush(stdout);
        DataSpace space({4, 4});
        println("PASS.");
        fflush(stdout);

        print("Test creating a dataset in a group... ");
        fflush(stdout);
        auto dataset = group.create_dataset<double>("test_data", space);
        println("PASS.");
        fflush(stdout);

        print("Test writing data to a dataset... ");
        fflush(stdout);
        dataset.write_data<double>(test_data.data());
        println("PASS.");
        fflush(stdout);

        print("Test writing an attribute to a group... ");
        fflush(stdout);
        group.write_attribute("test_attribute", test_attribute, {18});
        println("PASS.");
        fflush(stdout);

        print("Test writing an attribute to a dataset... ");
        fflush(stdout);
        dataset.write_attribute("test_attribute", test_attribute, {18});
        println("PASS.");
        fflush(stdout);

        print("Test opening an existing file... ");
        fflush(stdout);
        File test_file_2("test_file.hdf5", "r");
        println("PASS.");
        fflush(stdout);

        print("Test opening an existing group... ");
        fflush(stdout);
        Group group_2 = test_file.require_group("test_group");
        println("PASS.");
        fflush(stdout);

        print("Test opening an existing dataset... ");
        fflush(stdout);
        auto dataset_2 = group.open_dataset<double>("test_data");
        println("PASS.");
        fflush(stdout);

        print("Test reading data from a dataset... ");
        fflush(stdout);
        auto data = dataset_2.read_data<double>();
        for (hsize_t i = 0; i < dataset.size(); ++i)
        {
            if (test_data[i] == data[i])
            {
                continue;
            }
            else
            {
                println("FAIL! Data written and reread don't coincide!");
                fflush(stdout);
                return 1;
            }
        }
        print("PASS.");
        fflush(stdout);

        print("Test reading an attribute from a group... ");
        fflush(stdout);
        auto [att, d] = group_2.read_attribute<char>("test_attribute");

        if (not(strcmp(att.get(), test_attribute) == 0))
        {
            print("FAIL! Attribute written and reread don't coincide");
            fflush(stdout);
            return 1;
        }
        println("PASS.");
        fflush(stdout);

        print("Test reading an attribute from a dataset... ");
        fflush(stdout);
        auto [att_2, d_2] = dataset_2.read_attribute<char>("test_attribute");

        if (not(strcmp(att_2.get(), test_attribute) == 0))
        {
            println("FAIL! Attribute written and reread don't coincide!");
            fflush(stdout);
            return 1;
        }
        println("PASS.");
        fflush(stdout);
    }
    catch (runtime_error& e)
    {
        println("FAIL!!!!");
        println("An exception occurred: ");
        println("{}", e.what());
        fflush(stdout);
    }

    println("\nSuccess! All PASS!!");
    fflush(stdout);

    return 0;
}
