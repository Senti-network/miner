
if WANT_JANSSON
JANSSON_INCLUDES= -I$(top_srcdir)/compat/jansson
else
JANSSON_INCLUDES=
endif

EXTRA_DIST	= example-cfg.json nomacro.pl

SUBDIRS		= compat

CUSTATEVEC_INCLUDES = -I$(CUDA_PATH)/include -I$(CUQUANTUM_ROOT)/include
CUTENSORNET_INCLUDES = -I$(CUTENSORNET_ROOT)/include

ALL_INCLUDES	= @PTHREAD_FLAGS@ -fno-strict-aliasing $(JANSSON_INCLUDES) $(CUSTATEVEC_INCLUDES) $(CUTENSORNET_INCLUDES) -I.

bin_PROGRAMS	= qubitcoin-miner

dist_man_MANS	= cpuminer.1

qubitcoin_miner_SOURCES = \
  dummy.cpp \
  cpu-miner.c \
  util.c \
  api.c \
  sysinfos.c \
  algo-gate-api.c\
  malloc-huge.c \
  algo/keccak/sph_keccak.c \
  algo/keccak/keccak.c\
  algo/keccak/keccak-hash-4way.c \
  algo/keccak/keccak-4way.c\
  algo/keccak/keccak-gate.c \
  algo/keccak/sha3d-4way.c \
  algo/keccak/sha3d.c \
  algo/qhash/qhash-gate.c \
  algo/qhash/qhash.c \
  algo/qhash/qhash-custatevec.c \
  algo/qhash/qhash-cutensornet.c \
  algo/sha/sha1.c \
  algo/sha/sha1-hash.c \
  algo/sha/sha256-hash.c \
  algo/sha/sph_sha2.c \
  algo/sha/sph_sha2big.c \
  algo/sha/sha256-hash-4way.c \
  algo/sha/sha512-hash-4way.c \
  algo/sha/hmac-sha256-hash.c \
  algo/sha/hmac-sha256-hash-4way.c \
  algo/sha/sha256d.c \
  algo/sha/sha256d-4way.c \
  algo/sha/sha256t-gate.c \
  algo/sha/sha256t-4way.c \
  algo/sha/sha256q-4way.c \
  algo/sha/sha256q.c \
  algo/sha/sha512256d-4way.c \
  algo/sha/sha256dt.c \
  algo/simd/nist.c \
  algo/simd/vector.c \
  algo/simd/sph_simd.c \
  algo/simd/simd-hash-2way.c

disable_flags = -DNOASM

CUSTATEVEC_LDFLAGS = -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib -L$(CUQUANTUM_ROOT)/lib -lcustatevec_static -lcublas -lcudart # -lcustatevec_static -lcublas_static -lcublasLt_static -lculibos -lcudart_static

qubitcoin_miner_LDFLAGS	= @LDFLAGS@
CUTENSORNET_LDFLAGS = -L$(CUTENSORNET_ROOT)/lib -lcutensornet
qubitcoin_miner_LDADD	= @LIBCURL@ @JANSSON_LIBS@ @PTHREAD_LIBS@ @WS2_LIBS@ -lgmp $(CUSTATEVEC_LDFLAGS) $(CUTENSORNET_LDFLAGS)
qubitcoin_miner_CPPFLAGS = @LIBCURL_CPPFLAGS@ $(ALL_INCLUDES)
qubitcoin_miner_CFLAGS   = -Wno-pointer-sign -Wno-pointer-to-int-cast $(disable_flags) -g -funwind-tables -fno-omit-frame-pointer

# copy/paste from generated Makefile
common_ccflags = $(DEFS) $(ALL_INCLUDES) $(qubitcoin_miner_CPPFLAGS) $(CPPFLAGS) $(qubitcoin_miner_CFLAGS) $(CFLAGS)

