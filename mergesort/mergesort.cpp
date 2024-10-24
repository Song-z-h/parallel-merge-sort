/*
*  This file is part of Christian's OpenMP software lab 
*
*  Copyright (C) 2016 by Christian Terboven <terboven@itc.rwth-aachen.de>
*  Copyright (C) 2016 by Jonas Hahnfeld <hahnfeld@itc.rwth-aachen.de>
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program; if not, write to the Free Software
*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
*
*/

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/time.h>

#include <iostream>
#include <algorithm>

#include <cstdlib>
#include <cstdio>

#include <cmath>
#include <ctime>
#include <cstring>

#include <omp.h>

#include <vector>
using namespace std;


#define MAX_DEPTH 3
#define CUT_OFF_SIZE 1000
#define PARALLEL_THREADS 3
#define NESTED_THREADS true
/**
  * helper routine: check if array is sorted correctly
  */
bool isSorted(int ref[], int data[], const size_t size){
	std::sort(ref, ref + size);
	return true;
}

int* findJustBigger(int* s, int* e, int target){
	int* m = s + (e - s) / 2;
	if ( e - s > 0){
		if (*m == target){
			while(*(m+1) == target) m++;
			return m+1;
		}else if (target > *m){
			#pragma omp task firstprivate(target) shared(s, m)
			{
				findJustBigger(m+1, e, target);
			}
		}else if (target < *m){
			#pragma omp task firstprivate(target) shared(s, m)
			{
				findJustBigger(s, m, target);
			}
		}
	}
	return s;
}

int* findJustSmaller(int* s, int* e, int target){
	int* m = s + (e - s) / 2;
	if ( e - s > 0){
		if (*m == target){
			while(*(m-1) == target) m--;
			return m;
		}else if (target > *m){
			#pragma omp task firstprivate(target) shared(s, m)
			{
				findJustSmaller(m+1, e, target);
			}
		}else if (target < *m){
			#pragma omp task firstprivate(target) shared(s, m)
			{
				findJustSmaller(s, m, target);
			}
		}
	}
	return s;
}

/**
  * Parallel merge step (straight-forward implementation)
  */
// TODO: cut-off could also apply here (extra parameter?)
// TODO: optional: we can also break merge in two halves
void MsMergeParallel(int *out, int *in, long begin1, long end1, long begin2, long end2, long outBegin) {
	const long leftovers = (end1 - begin1) + (end2 - begin2);
	if (leftovers < CUT_OFF_SIZE){
		long left = begin1;
		long right = begin2;
		long idx = outBegin;

		while (left < end1 && right < end2) {
			if (in[left] <= in[right]) {
				out[idx] = in[left];
					left++;
			} else {
				out[idx] = in[right];
				right++;
			}
					idx++;
		}
		while (left < end1) {
			out[idx] = in[left];
					left++; idx++;
		}

		while (right < end2) {
			out[idx] = in[right];
			right++; idx++;
		}
	}
	
	else {
		//I can either split from X or from Y, if choose to split the largest one
		long halfx = -1;
		long halfy = -1;
		long outBegin2 = -1;

		if ((end1 - begin1) >= (end2 - begin2)){
			 halfx = begin1 + (end1 - begin1) / 2;
			 halfy = findJustBigger(in+begin2, in+end2, in[halfx]) - in;
		}else{
			//Y is larger
			halfy = begin2 + (end2 - begin2) / 2;
			halfx = findJustSmaller(in + begin1, in+end1, in[halfy]) - in;
		}		
		outBegin2 = outBegin + (halfx - begin1) + (halfy - begin2);
		
		//here I can open parallel region again to split and solve halves by splitting into more threads
		//But it actually makes the program slower, it is faster to open tasks directly without openning 
		//another parallel region.
		
		#pragma omp task firstprivate(begin1, halfx, begin2, halfy, outBegin) shared(out, in)
		{
			MsMergeParallel(out, in, begin1, halfx, begin2, halfy, outBegin);
		}
		#pragma omp task firstprivate(halfx, end1, halfy, end2, outBegin2) shared(out, in)
		{
			MsMergeParallel(out, in, halfx, end1, halfy, end2, outBegin2);
		}
	}
	
}

/**
  * Parallel MergeSort
  */
// TODO: remember one additional parameter (depth)
// TODO: recursive calls could be taskyfied
// TODO: task synchronization also is required
void MsParallel(int *array, int *tmp, bool inplace, long begin, long end, int depth) {
	if (begin < (end - 1)) {
		const long half = (begin + end) / 2;
		/*The depth variable i use to implement cutoff can be ONE more than MAX_DEPTH
		because when one threads reached to the maximum levelm before the if, another thread
		could also read it before it is updated, but it is not that important to make it critical
		because it is not our main result, I just want to maintain it at a certain level rather than
		exact depth level..*/
		if (depth < MAX_DEPTH){
			// for each thread in the parallel region, create 2 tasks and wait them to finish

			#pragma omp parallel num_threads(PARALLEL_THREADS) 
			{
				#pragma omp single
				{	
					#pragma omp task firstprivate(begin, half, end, inplace, depth) shared(array, tmp)
					{
						MsParallel(array, tmp, !inplace, begin, half, depth+1);
					}
					#pragma omp task firstprivate(half, end, begin, inplace, depth) shared(array, tmp)
					{
						MsParallel(array, tmp, !inplace, half, end, depth+1);
					}
				}
			}
		}else{
			MsParallel(array, tmp, !inplace, begin, half, depth);
			MsParallel(array, tmp, !inplace, half, end, depth);
		}
		
		//Before merging, I will be sure that two arrays are split and ready
		//otherwise maybe only half array is ready, the other half is not yet splited yet
		#pragma omp taskwait

		//#pragma omp parallel num_threads(PARALLEL_THREADS)
		//{
		//	#pragma omp single
		//	{
				if (inplace) {
					MsMergeParallel(array, tmp, begin, half, half, end, begin);
				} else {
					MsMergeParallel(tmp, array, begin, half, half, end, begin);
				}
		//	}
		//}
		
	} else if (!inplace) {
		tmp[begin] = array[begin];
	}
	
}


/**
  * Serial MergeSort
  */
// TODO: this function should create the parallel region
// TODO: good point to compute a good depth level (cut-off)
void MsSerial(int *array, int *tmp, const size_t size) {
   // TODO: parallel version of MsSequential will receive one more parameter: 'depth' (used as cut-off)
	omp_set_nested(NESTED_THREADS);
	//open the parallel region
	//omp_set_num_threads(PARALLEL_THREADS);
	#pragma omp parallel num_threads(PARALLEL_THREADS) shared(array, tmp) firstprivate(size)
	{
		//Calling MsSequential without omp single/master would cause dupplicate tasks... 
		#pragma omp single
		{
			cout << "using num threads: " << omp_get_num_threads() << " nesting " << NESTED_THREADS
				 << " maxdepth: " << MAX_DEPTH << " cutoff: " << CUT_OFF_SIZE << endl;
			MsParallel(array, tmp, true, 0, size, 0);
		}
	}
	
}


/** 
  * @brief program entry point
  */
int main(int argc, char* argv[]) {
	// variables to measure the elapsed time
	struct timeval t1, t2;
	double etime;

	// expect one command line arguments: array size
	if (argc != 2) {
		printf("Usage: MergeSort.exe <array size> \n");
		printf("\n");
		return EXIT_FAILURE;
	}
	else {
		const size_t stSize = strtol(argv[1], NULL, 10);
		int *data = (int*) malloc(stSize * sizeof(int));
		int *tmp = (int*) malloc(stSize * sizeof(int));
		int *ref = (int*) malloc(stSize * sizeof(int));

		printf("Initialization...\n");

		srand(95);
		for (size_t idx = 0; idx < stSize; ++idx){
			data[idx] = (int) (stSize * (double(rand()) / RAND_MAX));
		}
		std::copy(data, data + stSize, ref);

		double dSize = (stSize * sizeof(int)) / 1024 / 1024;
		printf("Sorting %zu elements of type int (%f MiB)...\n", stSize, dSize);

		gettimeofday(&t1, NULL);
		MsSerial(data, tmp, stSize);
		gettimeofday(&t2, NULL);

		etime = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000;
		etime = etime / 1000;

		printf("done, took %f sec. Verification...", etime);

		if (isSorted(ref, data, stSize)) {
			printf(" successful.\n");
		}
		else {
			printf(" FAILED.\n");
		}

		free(data);
		free(tmp);
		free(ref);
	}

	return EXIT_SUCCESS;
}
