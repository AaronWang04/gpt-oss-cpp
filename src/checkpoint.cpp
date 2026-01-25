/*
saftensor file format (predictably in this order)
```
{
  "TENSOR_NAME": {
    "dtype": "F16",
    "shape": [1, 16, 256],
    "data_offsets": [BEGIN, END]
  },
  "__metadata__": {
    "any_key": "any_string_value"
  }
}
```

*/
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include "utils.cpp"

enum DType {
	BF16, U8
};

struct TensorMeta_ {
	std::string name;
	DType dtype;
	std::vector<std::uint64_t> shape;
	std::vector<std::uint64_t> offset;
};

inline std::string to_string(DType d) {
    switch (d) {
        case DType::BF16:   return "DType::BF16";
        case DType::U8: return "DType::U8";
    }
    return "unknown";
}

inline std::string to_string(std::vector<std::uint64_t> vec) {
    std::stringstream ss;
    ss << '[';
    for (auto elm : vec) {
        ss << std::to_string(elm);
    }
    ss << ']';
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const TensorMeta_& T) {
	return os << T.name << ": " << to_string(T.dtype) << ", Shape: " << to_string(T.shape) << ", offset: " << to_string(T.offset) << '\n';
};


class Checkpoint {
private:
	std::uint64_t header_len;
	std::string header;
	std::vector<TensorMeta_> meta_;
	
public:
	Checkpoint(const std::string& path) {
		std::ifstream file_stream(path, std::ios::binary);
		if (!file_stream) throw std::runtime_error("wrong path or file DNE");

		try {
			file_stream.read(reinterpret_cast<char*>(&header_len), 8);
			header.resize(header_len);
			file_stream.read(&header[0], header_len);
		} catch (const std::exception& e) {
			std::cerr << e.what() << std::endl;
		}
		processHeader();
		file_stream.close();
		debugPrintCheckpoint();
	}

	void debugPrintCheckpoint() {
		for (auto tensor: meta_) {
			std::cout << tensor;
		}
	}

	void processHeader() {
		std::uint64_t i = 0;
		skipWhitespace(i);
		if (header[i] == '{') i++;
		else throw std::runtime_error("safetensor header unable to parse");

		while (i < header.size()) {
			skipWhitespace(i);
			if (header[i] == '}') break;
			TensorMeta_ meta_instance;
			std::string entry_name = parseJSONString(i);
			skipWhitespace(i);
			expectChar(i, ':');
			skipWhitespace(i);

			if (entry_name == "__metadata__") {
				skipJSONValue(i);
			} else {
				meta_instance.name = entry_name;
				parseTensorMeta(i, meta_instance);
				meta_.push_back(meta_instance);
			}

			skipWhitespace(i);
			if (header[i] == ',') {
				i++;
				continue;
			}
			if (header[i] == '}') break;
			throw std::runtime_error("safetensor header unable to parse");
		}

	}

	void skipWhitespace(std::uint64_t& i) {
		while (i < header.size() && std::isspace(static_cast<unsigned char>(header[i]))) i++;
	}

	void expectChar(std::uint64_t& i, char expected) {
		if (i >= header.size() || header[i] != expected) {
			throw std::runtime_error("safetensor header unable to parse");
		}
		i++;
	}

	std::string parseJSONString(std::uint64_t& i) {
		if (header[i] == '"') i++;
		else throw std::runtime_error("safetensor header unable to parse");
		
		std::string parsed_string;
		while (i < header.size()) {
			char c = header[i++];
			// assume no escape characters (who would put escape chars in a safetensor nani)
			if (c == '\\') {
				throw std::runtime_error("not implemented");
			} else if (c == '"') {
				break;
			} else {
				parsed_string.push_back(c);
			}
		}
		return parsed_string;
	}

	std::uint64_t parseUInt64(std::uint64_t& i) {
		skipWhitespace(i);
		if (i >= header.size() || !std::isdigit(static_cast<unsigned char>(header[i]))) {
			throw std::runtime_error("safetensor header unable to parse");
		}
		std::uint64_t value = 0;
		while (i < header.size() && std::isdigit(static_cast<unsigned char>(header[i]))) {
			value = value * 10 + static_cast<std::uint64_t>(header[i] - '0');
			i++;
		}
		return value;
	}

	std::vector<std::uint64_t> parseUInt64Array(std::uint64_t& i) {
		std::vector<std::uint64_t> values;
		expectChar(i, '[');
		skipWhitespace(i);
		if (header[i] == ']') {
			i++;
			return values;
		}
		while (i < header.size()) {
			values.push_back(parseUInt64(i));
			skipWhitespace(i);
			if (header[i] == ',') {
				i++;
				continue;
			}
			if (header[i] == ']') {
				i++;
				break;
			}
			throw std::runtime_error("safetensor header unable to parse");
		}
		return values;
	}

	DType parseDTypeString(const std::string& dtype) {
		if (dtype == "BF16") return DType::BF16;
		if (dtype == "U8") return DType::U8;
		throw std::runtime_error("unexpected dtype here");
	}

	void parseTensorMeta(std::uint64_t& i, TensorMeta_& meta_instance) {
		expectChar(i, '{');
		while (i < header.size()) {
			skipWhitespace(i);
			if (header[i] == '}') {
				i++;
				break;
			}
			std::string field = parseJSONString(i);
			skipWhitespace(i);
			expectChar(i, ':');
			skipWhitespace(i);

			if (field == "dtype") {
				meta_instance.dtype = parseDTypeString(parseJSONString(i));
			} else if (field == "shape") {
				meta_instance.shape = parseUInt64Array(i);
			} else if (field == "data_offsets") {
				meta_instance.offset = parseUInt64Array(i);
			} else {
				skipJSONValue(i);
			}

			skipWhitespace(i);
			if (header[i] == ',') {
				i++;
				continue;
			}
			if (header[i] == '}') {
				i++;
				break;
			}
			throw std::runtime_error("safetensor header unable to parse");
		}
	}

	void skipJSONValue(std::uint64_t& i) {
		skipWhitespace(i);
		if (i >= header.size()) throw std::runtime_error("safetensor header unable to parse");

		char c = header[i];
		if (c == '{') {
			i++;
			while (i < header.size()) {
				skipWhitespace(i);
				if (header[i] == '}') {
					i++;
					break;
				}
				parseJSONString(i);
				skipWhitespace(i);
				expectChar(i, ':');
				skipWhitespace(i);
				skipJSONValue(i);
				skipWhitespace(i);
				if (header[i] == ',') {
					i++;
					continue;
				}
				if (header[i] == '}') {
					i++;
					break;
				}
				throw std::runtime_error("safetensor header unable to parse");
			}
			return;
		}

		if (c == '[') {
			i++;
			while (i < header.size()) {
				skipWhitespace(i);
				if (header[i] == ']') {
					i++;
					break;
				}
				skipJSONValue(i);
				skipWhitespace(i);
				if (header[i] == ',') {
					i++;
					continue;
				}
				if (header[i] == ']') {
					i++;
					break;
				}
				throw std::runtime_error("safetensor header unable to parse");
			}
			return;
		}

		if (c == '"') {
			parseJSONString(i);
			return;
		}

		if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
			if (c == '-') i++;
			parseUInt64(i);
			return;
		}

		if (header.compare(i, 4, "true") == 0) {
			i += 4;
			return;
		}
		if (header.compare(i, 5, "false") == 0) {
			i += 5;
			return;
		}
		if (header.compare(i, 4, "null") == 0) {
			i += 4;
			return;
		}

		throw std::runtime_error("safetensor header unable to parse");
	}


};
