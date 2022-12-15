/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "profiler.h"

#include <unistd.h>
#include <sys/time.h>

double ncclProfilerFreq = -1;
uint64_t ncclProfilerClockStart;
void ncclProfilerClockCalibrate() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  uint64_t timeCycles = __rdtsc();
  double time = - tv.tv_sec*1E6 - tv.tv_usec;
  uint64_t total = 0ULL;
  for (int i=0; i<10000; i++) total += __rdtsc();
  gettimeofday(&tv, NULL);
  timeCycles = __rdtsc() - timeCycles;
  time += tv.tv_sec*1E6 + tv.tv_usec;
  ncclProfilerFreq = timeCycles/time;
  ncclProfilerClockStart = __rdtsc();
}

ncclProf_v1_t* ncclProfiler = NULL;
int ncclProfilerActive = 0;
struct ncclProfRecord* ncclProfilerRecord = NULL;
int ncclProfilerNrecs = 0;
#define NCCL_MAX_PROF_RECS 8
struct ncclProfRecord* ncclProfilerRecs[NCCL_MAX_PROF_RECS];

ncclResult_t ncclProfAddRec(struct ncclProfRecord** rec) {
  NCCLCHECK(ncclCalloc(rec, 1));
  ncclProfilerRecs[ncclProfilerNrecs++] = *rec;
  return ncclSuccess;
}

ncclResult_t ncclProfGetRecs(int* nRecs, struct ncclProfRecord*** recs) {
  *nRecs = ncclProfilerNrecs;
  *recs = ncclProfilerRecs;
  return ncclSuccess;
}

ncclResult_t ncclInternalProfilerDumpRecs(int nrecs, struct ncclProfRecord** recs) {
  if (nrecs == 0) return ncclSuccess;

  const char* str = getenv("NCCL_PROXY_PROFILE");
  if (!str) return ncclSuccess;
  FILE* f = fopen(str, "w");
  fprintf(f, "[\n");

  for (int r=0; r<nrecs; r++) {
    struct ncclProfRecord* rec = recs[r];
    for (int i=0; i<rec->nextId; i++) {
      struct ncclProfEvent* e = rec->events[ID_BLOCK(i)]+ID_IDX(i);
      fprintf(f, "{ \"name\": \"%s\", \"cat\": \"NET\", \"pid\": %d, \"tid\": %d, \"ts\": %f, \"id\": %d, \"ph\": \"b\", \"args\": { ", rec->typeNames[e->type], rec->rank, rec->rank, e->ts[NCCL_PROF_TS_START], i);
      for (int i=0; i<rec->kvSize; i++) {
        if (i) fprintf(f, ", ");
        fprintf(f, "\"%s\": ", rec->kvNames[i]);
        switch (rec->kvPrintFmt[i]) {
          case 'd': fprintf(f, "%ld", e->kv[i]); break;
          case 'u': fprintf(f, "%lu", e->kv[i]); break;
          case 'x': fprintf(f, "0x%lx", e->kv[i]); break;
          default:
                    printf("Unknown format %d\n", rec->kvPrintFmt[i]);
                    return ncclInternalError;
        }
      }
      fprintf(f, " } },\n");
      int startIndex = NCCL_PROF_TS_START;
      for (int tsIndex=2; tsIndex<NCCL_PROF_TS_SIZE; tsIndex++) {
        if (e->ts[tsIndex] == e->ts[NCCL_PROF_TS_START]) continue;
        fprintf(f, "{ \"name\": \"%s\", \"cat\": \"NET\", \"pid\": %d, \"tid\": %d, \"id\": %d, \"ts\": %f, \"ph\": \"%c\" },\n", rec->tsNames[tsIndex], rec->rank, rec->rank, i, e->ts[startIndex], 'b');
        fprintf(f, "{ \"name\": \"%s\", \"cat\": \"NET\", \"pid\": %d, \"tid\": %d, \"id\": %d, \"ts\": %f, \"ph\": \"%c\" },\n", rec->tsNames[tsIndex], rec->rank, rec->rank, i, e->ts[tsIndex], 'e');
        startIndex = tsIndex;
      }
      fprintf(f, "{ \"name\": \"%s\", \"cat\": \"NET\", \"pid\": %d, \"tid\": %d, \"ts\": %f, \"id\": %d, \"ph\": \"e\" },\n", rec->typeNames[e->type], rec->rank, rec->rank, e->ts[NCCL_PROF_TS_END], i);
    }
  }

  fprintf(f, "{} ]\n");
  fclose(f);
  return ncclSuccess;
}

ncclProfGetRecFn_t ncclInternalProfilerGetRec;
ncclResult_t ncclInternalProfilerInit(int* active, ncclProfGetRecFn_t getRec) {
  ncclInternalProfilerGetRec = getRec;
  char *ncclProfileEnv = getenv("NCCL_PROFILE");
  if (!ncclProfileEnv) {
    return ncclSuccess;
  }
  if (atoi(ncclProfileEnv) == 1) {
    *active = 1;
  }
  return ncclSuccess;
}

ncclResult_t ncclInternalProfilerEventStart(struct ncclProfRecord* rec, int* id, int type, int kvsize, va_list va) {
  uint64_t eventId = rec->nextId++;
  int eventBlock = ID_BLOCK(eventId);
  int eventIdx = ID_IDX(eventId);
  if (eventIdx == 0) {
    rec->events[eventBlock] = (struct ncclProfEvent*)malloc(sizeof(struct ncclProfEvent)*NCCL_PROF_MAX_EVENTS);
    if (rec->events[eventBlock] == NULL) return ncclSystemError;
  }

  struct ncclProfEvent* e = rec->events[eventBlock]+eventIdx;
  double time = profGettime();
  for (int i=0; i<NCCL_PROF_TS_SIZE; i++) {
    e->ts[i] = time;
  }
  e->type = type;
  for (int i=0; i<kvsize; i++) {
    e->kv[i] = va_arg(va, uint64_t);
  }
  *id = eventId;
  return ncclSuccess;
}

ncclResult_t ncclInternalProfilerEventTime(struct ncclProfRecord *rec, int id, int tsId) {
  rec->events[ID_BLOCK(id)][ID_IDX(id)].ts[tsId] = profGettime();
  return ncclSuccess;
}

ncclResult_t ncclInternalProfilerFreeRec(struct ncclProfRecord* rec) {
  for (int i=0; i<NCCL_PROF_MAX_EVENT_BLOCKS; i++) {
    free(rec->events[i]);
    rec->events[i] = NULL;
  }
  free(rec);
  return ncclSuccess;
}

ncclResult_t ncclInternalProfilerExit() {
  int nrecs;
  struct ncclProfRecord** recs;
  NCCLCHECK(ncclInternalProfilerGetRec(&nrecs, &recs));
  NCCLCHECK(ncclInternalProfilerDumpRecs(nrecs, recs));
  return ncclSuccess;
}

ncclProf_v1_t ncclInternalProfiler = {
  "Internal",
  ncclInternalProfilerInit,
  ncclInternalProfilerEventStart,
  ncclInternalProfilerEventTime,
  ncclInternalProfilerFreeRec,
  ncclInternalProfilerExit
};

ncclResult_t ncclProfInit() {
  void* profPluginLib = dlopen("libnccl-prof.so", RTLD_NOW | RTLD_LOCAL);
  if (profPluginLib) {
    INFO(NCCL_ALL, "Loaded external profiler");
    ncclProfiler = (ncclProf_v1_t*)dlsym(profPluginLib, "ncclProfPlugin_v1");

    if (!ncclProfiler) {
      WARN("Error loading ncclProfPlugin_v1 symbol from libnccl-prof.so");
      return ncclInternalError;
    }
  } else ncclProfiler = &ncclInternalProfiler;
  ncclProfiler->init(&ncclProfilerActive, ncclProfGetRecs);
  ncclProfilerClockCalibrate();
  return ncclSuccess;
}

ncclResult_t ncclProfStop() {
  ncclProfiler->exit();
  for (int r=0; r<ncclProfilerNrecs; r++) ncclProfFreeRec(ncclProfilerRecs[r]);
  return ncclSuccess;
}
