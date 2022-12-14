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
  if (ncclProfilerActive == 0) return ncclSuccess;

  uint64_t eventId = rec->nextId++;
  int eventBlock = ID_BLOCK(eventId);
  int eventIdx = ID_IDX(eventId);
  if (eventIdx == 0) {
    rec->events[eventBlock] = (struct ncclProfEvent*)malloc(sizeof(struct ncclProfEvent)*NCCL_PROF_MAX_EVENTS);
    if (rec->events[eventBlock] == NULL) return ncclSystemError;
  }

  struct ncclProfEvent* e = rec->events[eventBlock]+eventIdx;
  double time = profGettime();
  for (int i=0; i<NCCL_PROF_TS_SIZE; i++) e->ts[i] = time;
  e->type = type;
  va_list va;
  va_start(va, kvsize);
  for (int i=0; i<kvsize; i++) {
    e->kv[i] = va_arg(va, uint64_t);
  }
  va_end(va);
  *id = eventId;
  return ncclSuccess;
}
static void ncclProfEventTime(struct ncclProfRecord* rec, int id, int tsId) {
  if (ncclProfilerActive == 0) return;
  rec->events[ID_BLOCK(id)][ID_IDX(id)].ts[tsId] = profGettime();
}

static ncclResult_t ncclProfFreeRec(struct ncclProfRecord* rec) {
  for (int i=0; i<NCCL_PROF_MAX_EVENT_BLOCKS; i++) {
    free(rec->events[i]);
    rec->events[i] = NULL;
  }
  free(rec);
  return ncclSuccess;
}

ncclResult_t ncclProfInit();
ncclResult_t ncclProfStop();
#endif
