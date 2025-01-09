#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <semaphore.h>
#include <sys/wait.h>

#define NUM_CHILDREN 3
#define TERMS 10

void calculate_series(int id, sem_t *my_sem, sem_t *next_sem) {
    for (int k = 0; k < TERMS; k++) {
        sem_wait(my_sem); 
        printf("%d ", 3 * k + id);
        fflush(stdout);
        sem_post(next_sem);
    }
    exit(0);
}

int main() {
    pid_t pids[NUM_CHILDREN];
    sem_t *sem[NUM_CHILDREN];

    sem[0] = sem_open("/sem1", O_CREAT | O_EXCL, 0644, 1);
    sem[1] = sem_open("/sem2", O_CREAT | O_EXCL, 0644, 0);
    sem[2] = sem_open("/sem3", O_CREAT | O_EXCL, 0644, 0);

    if (sem[0] == SEM_FAILED || sem[1] == SEM_FAILED || sem[2] == SEM_FAILED) {
        perror("sem_open");
        exit(1);
    }

    for (int i = 0; i < NUM_CHILDREN; i++) {
        if ((pids[i] = fork()) == 0) {
            calculate_series(i + 1, sem[i], sem[(i + 1) % NUM_CHILDREN]);
        }
    }

    for (int i = 0; i < NUM_CHILDREN; i++) {
        waitpid(pids[i], NULL, 0);
    }

    sem_close(sem[0]);
    sem_close(sem[1]);
    sem_close(sem[2]);
    sem_unlink("/sem1");
    sem_unlink("/sem2");
    sem_unlink("/sem3");

    printf("\n");
    return 0;
}
