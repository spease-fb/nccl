/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PROFILER_H_
#define NCCL_PROFILER_H_

#include "proxy.h"

#define NCCL_PROF_MAX_EVENT_BLOCKS 1024
#define NCCL_PROF_TS_SIZE 8
#define NCCL_PROF_TS_START 0
#define NCCL_PROF_TS_END 1
#define NCCL_PROF_EVENT_KVSIZE 7

struct ncclProfEvent_v1 {
  double ts[NCCL_PROF_TS_SIZE];
  uint8_t type;
  uint8_t kvSize;
  uint64_t kv[NCCL_PROF_EVENT_KVSIZE];
};
#if __cplusplus >= 201703L
static_assert(sizeof(struct ncclProfEvent_v1) == 128);
#endif

struct ncclProfRecord_v1 {
  struct ncclProfEvent_v1 *events[NCCL_PROF_MAX_EVENT_BLOCKS];
  int rank;
  uint64_t nextId; // Used internally
  const char** typeNames;
  const char** tsNames;
  int kvSize;
  const char** kvNames;
  const char* kvPrintFmt;
};

typedef ncclResult_t (*ncclProfGetRecFn_t)(int* nRecs, struct ncclProfRecord_v1*** recs);

typedef struct {
  const char* name;
  ncclResult_t (*init)(int* active, ncclProfGetRecFn_t getRec);
  ncclResult_t (*eventStart)(struct ncclProfRecord_v1 *rec, int *id, int type, int kvsize, va_list args);
  ncclResult_t (*eventTime)(struct ncclProfRecord_v1 *rec, int id, int tsId);
  ncclResult_t (*freeRec)(struct ncclProfRecord_v1 *rec);
  ncclResult_t (*exit)();
}ncclProf_v1_t;

extern ncclProf_v1_t* ncclProfiler;
extern int ncclProfilerActive;

#define ncclProfEvent ncclProfEvent_v1
#define ncclProfRecord ncclProfRecord_v1

#include <x86intrin.h>
extern double ncclProfilerFreq;
extern uint64_t ncclProfilerClockStart;
static inline double profGettime() {
  return (__rdtsc()-ncclProfilerClockStart)/ncclProfilerFreq;
}

#define NCCL_PROF_MAX_EVENTS 65536
#define ID_BLOCK(id) (id>>16)
#define ID_IDX(id) (id&0xffff)

ncclResult_t ncclProfAddRec(struct ncclProfRecord** rec);

#include <stdarg.h>

static ncclResult_t ncclProfEventStart(struct ncclProfRecord* rec, int* id, int type, int kvsize, ...) {
  if (ncclProfilerActive == 0) {
    return ncclSuccess;
  }

  va_list args;
  va_start(args, kvsize);
  const ncclResult_t r = ncclProfiler->eventStart(rec, id, type, kvsize, args);
  va_end(args);

  return r;
}

static ncclResult_t ncclProfEventTime(struct ncclProfRecord* rec, int id, int tsId) {
  if (ncclProfilerActive == 0) {
    return ncclSuccess;
  }

  return ncclProfiler->eventTime(rec, id, tsId);
}

static ncclResult_t ncclProfFreeRec(struct ncclProfRecord* rec) {
  if (ncclProfilerActive == 0) {
    return ncclSuccess;
  }

  return ncclProfiler->freeRec(rec);
}

ncclResult_t ncclProfInit();
ncclResult_t ncclProfStop();
#endif
