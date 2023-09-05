CXXFLAGS += -O3 -Wall -shared -std=c++11 -fPIC -fdiagnostics-color
CPPFLAGS += $(shell /raid/s3/opengptx/alexw/conda_venvs/megatron_lm/bin/python -m pybind11 --includes)
LIBNAME = helpers
LIBEXT = $(shell python3-config --extension-suffix)

default: $(LIBNAME)$(LIBEXT)

%$(LIBEXT): %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
