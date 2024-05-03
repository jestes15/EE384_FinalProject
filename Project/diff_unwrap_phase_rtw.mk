###########################################################################
## Makefile generated for component 'diff_unwrap_phase'. 
## 
## Makefile     : diff_unwrap_phase_rtw.mk
## Generated on : Fri May 03 13:01:32 2024
## Final product: .\diff_unwrap_phase.lib
## Product type : static-library
## 
###########################################################################

###########################################################################
## MACROS
###########################################################################

# Macro Descriptions:
# PRODUCT_NAME            Name of the system to build
# MAKEFILE                Name of this makefile
# MODELLIB                Static library target

PRODUCT_NAME              = diff_unwrap_phase
MAKEFILE                  = diff_unwrap_phase_rtw.mk
MATLAB_ROOT               = C:\PROGRA~1\MATLAB\R2023b
MATLAB_BIN                = C:\PROGRA~1\MATLAB\R2023b\bin
MATLAB_ARCH_BIN           = $(MATLAB_BIN)\win64
START_DIR                 = C:\Users\sagep\OneDrive\Documents\MATLAB\UAH
TGT_FCN_LIB               = ISO_C++11
SOLVER_OBJ                = 
CLASSIC_INTERFACE         = 0
MODEL_HAS_DYNAMICALLY_LOADED_SFCNS = 
RELATIVE_PATH_TO_ANCHOR   = ..\..\..
C_STANDARD_OPTS           = 
CPP_STANDARD_OPTS         = 
NODEBUG                   = 1
MODELLIB                  = diff_unwrap_phase.lib

###########################################################################
## TOOLCHAIN SPECIFICATIONS
###########################################################################

# Toolchain Name:          NVIDIA CUDA (w/Microsoft Visual C++ 2022) | nmake (64-bit Windows)
# Supported Version(s):    17.0
# ToolchainInfo Version:   2023b
# Specification Revision:  1.0
# 
#-------------------------------------------
# Macros assumed to be defined elsewhere
#-------------------------------------------

# NODEBUG
# cvarsdll
# cvarsmt
# conlibsmt
# ldebug
# conflags
# cflags
# C_STANDARD_OPTS
# CPP_STANDARD_OPTS

#-----------
# MACROS
#-----------

MEX_OPTS_FILE      = $(MATLAB_ROOT)\bin\$(ARCH)\mexopts\msvc2022.xml
MW_EXTERNLIB_DIR   = $(MATLAB_ROOT)\extern\lib\win64\microsoft
MW_LIB_DIR         = $(MATLAB_ROOT)\lib\win64
WARN_FLAGS         = -Wall -W -Wwrite-strings -Winline -Wstrict-prototypes -Wnested-externs -Wpointer-arith -Wcast-align
WARN_FLAGS_MAX     = $(WARN_FLAGS) -Wcast-qual -Wshadow
CPP_WARN_FLAGS     = -Wall -W -Wwrite-strings -Winline -Wpointer-arith -Wcast-align
CPP_WARN_FLAGS_MAX = $(CPP_WARN_FLAGS) -Wcast-qual -Wshadow

TOOLCHAIN_SRCS = 
TOOLCHAIN_INCS = 
TOOLCHAIN_LIBS = 

#------------------------
# BUILD TOOL COMMANDS
#------------------------

# C Compiler: NVIDIA CUDA C Compiler Driver
CC = nvcc

# Linker: NVIDIA CUDA C Compiler Driver
LD = nvcc

# C++ Compiler: NVIDIA CUDA C++ Compiler Driver
CPP = nvcc

# C++ Linker: NVIDIA CUDA C++ Compiler Driver
CPP_LD = nvcc

# Archiver: Microsoft Visual C/C++ Archiver
AR = lib

# MEX Tool: MEX Tool
MEX_PATH = $(MATLAB_ARCH_BIN)
MEX = "$(MEX_PATH)\mex"

# Download: Download
DOWNLOAD =

# Execute: Execute
EXECUTE = $(PRODUCT)

# Builder: NMAKE Utility
MAKE = nmake


#-------------------------
# Directives/Utilities
#-------------------------

CDEBUG              = -g -G 
C_OUTPUT_FLAG       = -o 
LDDEBUG             = -g -G 
OUTPUT_FLAG         = -o 
CPPDEBUG            = -g -G 
CPP_OUTPUT_FLAG     = -o 
CPPLDDEBUG          = -g -G 
OUTPUT_FLAG         = -o 
ARDEBUG             =
STATICLIB_OUTPUT_FLAG = -out:
MEX_DEBUG           = -g
RM                  = @del
ECHO                = @echo
MV                  = @ren
RUN                 = @cmd /C

#--------------------------------------
# "Faster Runs" Build Configuration
#--------------------------------------

ARFLAGS              =
CFLAGS               = -c $(C_STANDARD_OPTS) -Wno-deprecated-gpu-targets -Xcompiler "/wd 4819" -rdc=true -Xcudafe "--display_error_number --diag_suppress=unsigned_compare_with_zero"  \
                       -O3
CPPFLAGS             = -c $(CPP_STANDARD_OPTS) -Wno-deprecated-gpu-targets -Xcompiler "/wd 4819" -rdc=true -Xcudafe "--display_error_number --diag_suppress=unsigned_compare_with_zero"  \
                       -O3
CPP_LDFLAGS          = -Xnvlink -w -Xarchive "/IGNORE:4006" -Xarchive "/IGNORE:4221" $(conlibs) cudart.lib -Wno-deprecated-gpu-targets
CPP_SHAREDLIB_LDFLAGS  = -shared -Xnvlink -w -Xarchive "/IGNORE:4006" -Xarchive "/IGNORE:4221" cudart.lib -Wno-deprecated-gpu-targets -Xlinker -dll -Xlinker -def:$(DEF_FILE)
DOWNLOAD_FLAGS       =
EXECUTE_FLAGS        =
LDFLAGS              = -Xnvlink -w -Xarchive "/IGNORE:4006" -Xarchive "/IGNORE:4221" $(conlibs) cudart.lib -Wno-deprecated-gpu-targets
MEX_CPPFLAGS         =
MEX_CPPLDFLAGS       =
MEX_CFLAGS           = -MATLAB_ARCH=$(ARCH) $(INCLUDES) \
                         \
                       COPTIMFLAGS="$(C_STANDARD_OPTS)  \
                       -O3 \
                        $(DEFINES)" \
                         \
                       -silent
MEX_LDFLAGS          = LDFLAGS=='$$LDFLAGS'
MAKE_FLAGS           = -f $(MAKEFILE)
SHAREDLIB_LDFLAGS    = -shared -Xnvlink -w -Xarchive "/IGNORE:4006" -Xarchive "/IGNORE:4221" cudart.lib -Wno-deprecated-gpu-targets -Xlinker -dll -Xlinker -def:$(DEF_FILE)



###########################################################################
## OUTPUT INFO
###########################################################################

PRODUCT = .\diff_unwrap_phase.lib
PRODUCT_TYPE = "static-library"
BUILD_TYPE = "Static Library"

###########################################################################
## INCLUDE PATHS
###########################################################################

INCLUDES_BUILDINFO = $(START_DIR)\codegen\lib\diff_unwrap_phase;$(START_DIR);$(MATLAB_ROOT)\extern\include

INCLUDES = $(INCLUDES_BUILDINFO)

###########################################################################
## DEFINES
###########################################################################

DEFINES_ = -D MW_CUDA_ARCH=350
DEFINES_CUSTOM = 
DEFINES_STANDARD = -D MODEL=diff_unwrap_phase

DEFINES = $(DEFINES_) $(DEFINES_CUSTOM) $(DEFINES_STANDARD)

###########################################################################
## SOURCE FILES
###########################################################################

SRCS = $(START_DIR)\codegen\lib\diff_unwrap_phase\diff_unwrap_phase_data.cu $(START_DIR)\codegen\lib\diff_unwrap_phase\rt_nonfinite.cu $(START_DIR)\codegen\lib\diff_unwrap_phase\rtGetNaN.cu $(START_DIR)\codegen\lib\diff_unwrap_phase\rtGetInf.cu $(START_DIR)\codegen\lib\diff_unwrap_phase\diff_unwrap_phase_initialize.cu $(START_DIR)\codegen\lib\diff_unwrap_phase\diff_unwrap_phase_terminate.cu $(START_DIR)\codegen\lib\diff_unwrap_phase\diff_unwrap_phase.cu $(START_DIR)\codegen\lib\diff_unwrap_phase\diff_unwrap_phase_emxutil.cu $(START_DIR)\codegen\lib\diff_unwrap_phase\diff_unwrap_phase_emxAPI.cu

ALL_SRCS = $(SRCS)

###########################################################################
## OBJECTS
###########################################################################

OBJS = diff_unwrap_phase_data.obj rt_nonfinite.obj rtGetNaN.obj rtGetInf.obj diff_unwrap_phase_initialize.obj diff_unwrap_phase_terminate.obj diff_unwrap_phase.obj diff_unwrap_phase_emxutil.obj diff_unwrap_phase_emxAPI.obj

ALL_OBJS = $(OBJS)

###########################################################################
## PREBUILT OBJECT FILES
###########################################################################

PREBUILT_OBJS = 

###########################################################################
## LIBRARIES
###########################################################################

LIBS = 

###########################################################################
## SYSTEM LIBRARIES
###########################################################################

SYSTEM_LIBS = 

###########################################################################
## ADDITIONAL TOOLCHAIN FLAGS
###########################################################################

#---------------
# C Compiler
#---------------

CFLAGS_ = -Xcompiler "/source-charset:utf-8"
CFLAGS_CU_OPTS = sm_89
CFLAGS_BASIC = $(DEFINES) 

CFLAGS = $(CFLAGS) $(CFLAGS_) $(CFLAGS_CU_OPTS) $(CFLAGS_BASIC)

#-----------------
# C++ Compiler
#-----------------

CPPFLAGS_ = -Xcompiler "/source-charset:utf-8"
CPPFLAGS_CU_OPTS = sm_89
CPPFLAGS_BASIC = $(DEFINES) 

CPPFLAGS = $(CPPFLAGS) $(CPPFLAGS_) $(CPPFLAGS_CU_OPTS) $(CPPFLAGS_BASIC)

#---------------
# C++ Linker
#---------------

CPP_LDFLAGS_ = sm_89 -lcublas -lcusolver -lcufft -lcurand -lcusparse

CPP_LDFLAGS = $(CPP_LDFLAGS) $(CPP_LDFLAGS_)

#------------------------------
# C++ Shared Library Linker
#------------------------------

CPP_SHAREDLIB_LDFLAGS_ = sm_89 -lcublas -lcusolver -lcufft -lcurand -lcusparse

CPP_SHAREDLIB_LDFLAGS = $(CPP_SHAREDLIB_LDFLAGS) $(CPP_SHAREDLIB_LDFLAGS_)

#-----------
# Linker
#-----------

LDFLAGS_ = sm_89 -lcublas -lcusolver -lcufft -lcurand -lcusparse

LDFLAGS = $(LDFLAGS) $(LDFLAGS_)

#--------------------------
# Shared Library Linker
#--------------------------

SHAREDLIB_LDFLAGS_ = sm_89 -lcublas -lcusolver -lcufft -lcurand -lcusparse

SHAREDLIB_LDFLAGS = $(SHAREDLIB_LDFLAGS) $(SHAREDLIB_LDFLAGS_)

###########################################################################
## INLINED COMMANDS
###########################################################################


!include $(MATLAB_ROOT)\rtw\c\tools\vcdefs.mak
.SUFFIXES: .cu


###########################################################################
## PHONY TARGETS
###########################################################################

.PHONY : all build clean info prebuild download execute set_environment_variables


all : build
	@cmd /C "@echo ### Successfully generated all binary outputs."


build : set_environment_variables prebuild $(PRODUCT)


prebuild : 


download : $(PRODUCT)


execute : download


set_environment_variables : 
	@set INCLUDE=$(INCLUDES);$(INCLUDE)
	@set LIB=$(LIB)


###########################################################################
## FINAL TARGET
###########################################################################

#---------------------------------
# Create a static library         
#---------------------------------

$(PRODUCT) : $(OBJS) $(PREBUILT_OBJS)
	@cmd /C "@echo ### Creating static library "$(PRODUCT)" ..."
	$(AR) $(ARFLAGS) -out:$(PRODUCT) $(OBJS)
	@cmd /C "@echo ### Created: $(PRODUCT)"


###########################################################################
## INTERMEDIATE TARGETS
###########################################################################

#---------------------
# SOURCE-TO-OBJECT
#---------------------

.cu.obj :
	$(CC) $(CFLAGS) -o  "$@" "$<"


.c.obj :
	$(CC) $(CFLAGS) -o  "$@" "$<"


.cu.obj :
	$(CPP) $(CPPFLAGS) -o  "$@" "$<"


.cpp.obj :
	$(CPP) $(CPPFLAGS) -o  "$@" "$<"


{$(RELATIVE_PATH_TO_ANCHOR)}.cu.obj :
	$(CC) $(CFLAGS) -o  "$@" "$<"


{$(RELATIVE_PATH_TO_ANCHOR)}.c.obj :
	$(CC) $(CFLAGS) -o  "$@" "$<"


{$(RELATIVE_PATH_TO_ANCHOR)}.cu.obj :
	$(CPP) $(CPPFLAGS) -o  "$@" "$<"


{$(RELATIVE_PATH_TO_ANCHOR)}.cpp.obj :
	$(CPP) $(CPPFLAGS) -o  "$@" "$<"


{$(START_DIR)\codegen\lib\diff_unwrap_phase}.cu.obj :
	$(CC) $(CFLAGS) -o  "$@" "$<"


{$(START_DIR)\codegen\lib\diff_unwrap_phase}.c.obj :
	$(CC) $(CFLAGS) -o  "$@" "$<"


{$(START_DIR)\codegen\lib\diff_unwrap_phase}.cu.obj :
	$(CPP) $(CPPFLAGS) -o  "$@" "$<"


{$(START_DIR)\codegen\lib\diff_unwrap_phase}.cpp.obj :
	$(CPP) $(CPPFLAGS) -o  "$@" "$<"


{$(START_DIR)}.cu.obj :
	$(CC) $(CFLAGS) -o  "$@" "$<"


{$(START_DIR)}.c.obj :
	$(CC) $(CFLAGS) -o  "$@" "$<"


{$(START_DIR)}.cu.obj :
	$(CPP) $(CPPFLAGS) -o  "$@" "$<"


{$(START_DIR)}.cpp.obj :
	$(CPP) $(CPPFLAGS) -o  "$@" "$<"


diff_unwrap_phase_data.obj : "$(START_DIR)\codegen\lib\diff_unwrap_phase\diff_unwrap_phase_data.cu"
	$(CPP) $(CPPFLAGS) -o  "$@" "$(START_DIR)\codegen\lib\diff_unwrap_phase\diff_unwrap_phase_data.cu"


rt_nonfinite.obj : "$(START_DIR)\codegen\lib\diff_unwrap_phase\rt_nonfinite.cu"
	$(CPP) $(CPPFLAGS) -o  "$@" "$(START_DIR)\codegen\lib\diff_unwrap_phase\rt_nonfinite.cu"


rtGetNaN.obj : "$(START_DIR)\codegen\lib\diff_unwrap_phase\rtGetNaN.cu"
	$(CPP) $(CPPFLAGS) -o  "$@" "$(START_DIR)\codegen\lib\diff_unwrap_phase\rtGetNaN.cu"


rtGetInf.obj : "$(START_DIR)\codegen\lib\diff_unwrap_phase\rtGetInf.cu"
	$(CPP) $(CPPFLAGS) -o  "$@" "$(START_DIR)\codegen\lib\diff_unwrap_phase\rtGetInf.cu"


diff_unwrap_phase_initialize.obj : "$(START_DIR)\codegen\lib\diff_unwrap_phase\diff_unwrap_phase_initialize.cu"
	$(CPP) $(CPPFLAGS) -o  "$@" "$(START_DIR)\codegen\lib\diff_unwrap_phase\diff_unwrap_phase_initialize.cu"


diff_unwrap_phase_terminate.obj : "$(START_DIR)\codegen\lib\diff_unwrap_phase\diff_unwrap_phase_terminate.cu"
	$(CPP) $(CPPFLAGS) -o  "$@" "$(START_DIR)\codegen\lib\diff_unwrap_phase\diff_unwrap_phase_terminate.cu"


diff_unwrap_phase.obj : "$(START_DIR)\codegen\lib\diff_unwrap_phase\diff_unwrap_phase.cu"
	$(CPP) $(CPPFLAGS) -o  "$@" "$(START_DIR)\codegen\lib\diff_unwrap_phase\diff_unwrap_phase.cu"


diff_unwrap_phase_emxutil.obj : "$(START_DIR)\codegen\lib\diff_unwrap_phase\diff_unwrap_phase_emxutil.cu"
	$(CPP) $(CPPFLAGS) -o  "$@" "$(START_DIR)\codegen\lib\diff_unwrap_phase\diff_unwrap_phase_emxutil.cu"


diff_unwrap_phase_emxAPI.obj : "$(START_DIR)\codegen\lib\diff_unwrap_phase\diff_unwrap_phase_emxAPI.cu"
	$(CPP) $(CPPFLAGS) -o  "$@" "$(START_DIR)\codegen\lib\diff_unwrap_phase\diff_unwrap_phase_emxAPI.cu"


###########################################################################
## DEPENDENCIES
###########################################################################

$(ALL_OBJS) : rtw_proj.tmw $(MAKEFILE)


###########################################################################
## MISCELLANEOUS TARGETS
###########################################################################

info : 
	@cmd /C "@echo ### PRODUCT = $(PRODUCT)"
	@cmd /C "@echo ### PRODUCT_TYPE = $(PRODUCT_TYPE)"
	@cmd /C "@echo ### BUILD_TYPE = $(BUILD_TYPE)"
	@cmd /C "@echo ### INCLUDES = $(INCLUDES)"
	@cmd /C "@echo ### DEFINES = $(DEFINES)"
	@cmd /C "@echo ### ALL_SRCS = $(ALL_SRCS)"
	@cmd /C "@echo ### ALL_OBJS = $(ALL_OBJS)"
	@cmd /C "@echo ### LIBS = $(LIBS)"
	@cmd /C "@echo ### MODELREF_LIBS = $(MODELREF_LIBS)"
	@cmd /C "@echo ### SYSTEM_LIBS = $(SYSTEM_LIBS)"
	@cmd /C "@echo ### TOOLCHAIN_LIBS = $(TOOLCHAIN_LIBS)"
	@cmd /C "@echo ### CFLAGS = $(CFLAGS)"
	@cmd /C "@echo ### LDFLAGS = $(LDFLAGS)"
	@cmd /C "@echo ### SHAREDLIB_LDFLAGS = $(SHAREDLIB_LDFLAGS)"
	@cmd /C "@echo ### CPPFLAGS = $(CPPFLAGS)"
	@cmd /C "@echo ### CPP_LDFLAGS = $(CPP_LDFLAGS)"
	@cmd /C "@echo ### CPP_SHAREDLIB_LDFLAGS = $(CPP_SHAREDLIB_LDFLAGS)"
	@cmd /C "@echo ### ARFLAGS = $(ARFLAGS)"
	@cmd /C "@echo ### MEX_CFLAGS = $(MEX_CFLAGS)"
	@cmd /C "@echo ### MEX_CPPFLAGS = $(MEX_CPPFLAGS)"
	@cmd /C "@echo ### MEX_LDFLAGS = $(MEX_LDFLAGS)"
	@cmd /C "@echo ### MEX_CPPLDFLAGS = $(MEX_CPPLDFLAGS)"
	@cmd /C "@echo ### DOWNLOAD_FLAGS = $(DOWNLOAD_FLAGS)"
	@cmd /C "@echo ### EXECUTE_FLAGS = $(EXECUTE_FLAGS)"
	@cmd /C "@echo ### MAKE_FLAGS = $(MAKE_FLAGS)"


clean : 
	$(ECHO) "### Deleting all derived files ..."
	@if exist $(PRODUCT) $(RM) $(PRODUCT)
	$(RM) $(ALL_OBJS)
	$(ECHO) "### Deleted all derived files."


