
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <semaphore.h>
#include <fcntl.h>

int isPrime(int num) {
    if (num <= 1) return 0;
    for (int i = 2; i * i <= num; i++) {
        if (num % i == 0) return 0;
    }
    return 1;
}

void printPrimes(int start, int end) {
    for (int i = start; i <= end; i++) {
        if (isPrime(i))
            printf("%d ", i);
    }
    printf("\n");
}

int main() {
    sem_t *sem = sem_open("/primeSem", O_CREAT, 0644, 1);

    if (fork() == 0) {
        sem_wait(sem);
        printf("Child 1: ");
        printPrimes(1, 100);
        sem_post(sem);
        exit(0);
    }

    if (fork() == 0) {
        sem_wait(sem);
        printf("Child 2: ");
        printPrimes(101, 200);
        sem_post(sem);
        exit(0);
    }

    if (fork() == 0) {
        sem_wait(sem);
        printf("Child 3: ");
        printPrimes(201, 300);
        sem_post(sem);
        exit(0);
    }

    for (int i = 0; i < 3; i++) {
        wait(NULL);
    }

    sem_close(sem);
    sem_unlink("/primeSem");

    return 0;
}
