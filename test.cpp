#include <iostream>
#include <cassert>
#include <cstring>
#include <cstdlib>
#include <ctime>

#include "headache5.hpp"


using namespace std;
using namespace H5;


int main()
{
    cout << "Generating random data for testing... " << flush;

    srand(time(0));

    vector<double> test_data;
    for (int i = 0; i < 16; ++i)
    {
        // test_data.push_back(static_cast<double>(rand()) / RAND_MAX * 10.);
        test_data.push_back(static_cast<double>(i));
    }

    const char test_attribute[] = "Greetings, human.";

    cout << "done." << endl;


    ////////////////////////////////////////////////////////////////////////////

    try
    {
        cout << "Test opening a new file... " << flush;
        File test_file("test_file.hdf5", "w");
        cout << "PASS." << endl;

        cout << "Test creating a group... " << flush;
        Group group = test_file.require_group("test_group");
        cout << "PASS." << endl;

        cout << "Test creating a dataspace... " << flush;
        DataSpace space({4, 4});
        cout << "PASS." << endl;

        cout << "Test creating a dataset in a group... " << flush;
        auto dataset = group.create_dataset<double>("test_data", space);
        cout << "PASS." << endl;

        cout << "Test writing data to a dataset... " << flush;
        dataset.write_data<double>(test_data.data());
        cout << "PASS." << endl;

        cout << "Test writing an attribute to a group... " << flush;
        group.write_attribute("test_attribute", test_attribute, {18});
        cout << "PASS." << endl;

        cout << "Test writing an attribute to a dataset... " << flush;
        dataset.write_attribute("test_attribute", test_attribute, {18});
        cout << "PASS." << endl;

        cout << "Test opening an existing file... " << flush;
        File test_file_2("test_file.hdf5", "r");
        cout << "PASS." << endl;

        cout << "Test opening an existing group... " << flush;
        Group group_2 = test_file.require_group("test_group");
        cout << "PASS." << endl;

        cout << "Test opening an existing dataset... " << flush;
        auto dataset_2 = group.open_dataset<double>("test_data");
        cout << "PASS." << endl;

        cout << "Test reading data from a dataset... " << flush;
        auto data = dataset_2.read_data<double>();
        for (hsize_t i = 0; i < dataset.size(); ++i)
        {
            if (test_data[i] == data[i])
            {
                continue;
            }
            else
            {
                cout << "FAIL! Data written and reread don't coincide!" << endl;
                return 1;
            }
        }
        cout << "PASS." << endl;

        cout << "Test reading an attribute from a group... " << flush;
        auto [att, d] = group_2.read_attribute<char>("test_attribute");

        if (not (strcmp(att.get(), test_attribute) == 0))
        {
            cout << "FAIL! Attribute written and reread don't coincide!"
                 << endl;
            return 1;
        }
        cout << "PASS." << endl;

        cout << "Test reading an attribute from a dataset... " << flush;
        auto [att_2, d_2] = dataset_2.read_attribute<char>("test_attribute");

        if (not (strcmp(att_2.get(), test_attribute) == 0))
        {
            cout << "FAIL! Attribute written and reread don't coincide!"
                 << endl;
            return 1;
        }
        cout << "PASS." << endl;
    }
    catch (runtime_error& e)
    {
        cout << "FAIL!!!!" << endl;
        cout << "An exception occurred: " << endl;
        cout << e.what() << endl;
    }

    cout << endl << "Success! All PASS!!" << endl;

    return 0;
}
