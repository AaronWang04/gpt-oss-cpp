#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../src/checkpoint.cpp"

int main() {
    try {
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "checkpoint tests failed: " << e.what() << std::endl;
        return 1;
    }
}

