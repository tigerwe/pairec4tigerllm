// block_consumer.c — 拦截 libdatasystem.so 的 consumer 线程创建
// 编译: gcc -shared -fPIC -o block_ds_consumer.so block_consumer.c -ldl
// 用法: LD_PRELOAD=./block_ds_consumer.so:/path/to/libabseil_dll.so...

#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <string.h>

static int (*real_pthread_create)(pthread_t *, const pthread_attr_t *,
                                   void *(*)(void *), void *) = NULL;

int pthread_create(pthread_t *thread, const pthread_attr_t *attr,
                   void *(*start_routine)(void *), void *arg)
{
    Dl_info info;
    if (!real_pthread_create)
        real_pthread_create = (typeof(real_pthread_create))dlsym(RTLD_NEXT, "pthread_create");

    if (dladdr((void *)start_routine, &info) && info.dli_fname &&
        strstr(info.dli_fname, "libdatasystem"))
    {
        // 假装成功，不创建线程，避免 consumer 线程访问未映射的共享内存
        *thread = 0;
        return 0;
    }
    return real_pthread_create(thread, attr, start_routine, arg);
}
