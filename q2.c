
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();

    if (pid == 0) { 
        // Child process
        FILE *file = fopen("data.txt", "r");
        if (file == NULL) {
            printf("Error opening file.\n");
            return 1;
        }

        int sum = 0, num;
        while (fscanf(file, "%d", &num) != EOF) {
            sum += num;
        }

        fclose(file);
        printf("Sum of integers in the file: %d\n", sum);
    } else if (pid > 0) { 
        // Parent process
        wait(NULL); // Wait for the child process to finish
    } else { 
        printf("Fork failed.\n");
        return 1;
    }

    return 0;
}
