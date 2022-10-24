#pragma once

#include <vector>
#include <bitset>
#include <iostream>
void test(void); 


class CPPGroup
{
public:
    
    std::vector<short> gvec;
    int size_; 
    //std::bitset<4096> _inset; 
    CPPGroup() :
        size_(0)
        ///_inset(4096)
    {
        gvec.reserve(10);


    }

    ~CPPGroup() {
        //std::cout << "this group was " << size_ << "elements" << std::endl; 
    }
    int contains(int x ) {

        
        for(int i = 0; i < gvec.size(); ++i) {
            if (x == gvec[i]) {
                return true;
            } 
        }
        return false; 
        //return _inset.test(x);
    }

    void insert(int x) {
        if(!contains(x)) {
            gvec.push_back(x);
            size_++; 
            //_inset.set(x);// = 1; 
        }


    }
}; 
