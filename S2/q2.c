#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <ctype.h>

#define FILENAME "input.txt"

void count_words() {
    FILE *file = fopen(FILENAME, "r");
    if (!file) {
        perror("Error opening file");
        exit(1);
    }
    int words = 0, in_word = 0;
    char ch;

    while ((ch = fgetc(file)) != EOF) {
        if (isspace(ch)) {
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            words++;
        }
    }

    fclose(file);
    printf("Number of words: %d\n", words);
    exit(0);
}

void count_lines() {
    FILE *file = fopen(FILENAME, "r");
    if (!file) {
        perror("Error opening file");
        exit(1);
    }
    int lines = 0;
    char ch;

    while ((ch = fgetc(file)) != EOF) {
        if (ch == '\n') {
            lines++;
        }
    }

    fclose(file);
    printf("Number of lines: %d\n", lines);
    exit(0);
}

void count_characters() {
    FILE *file = fopen(FILENAME, "r");
    if (!file) {
        perror("Error opening file");
        exit(1);
    }
    int characters = 0;

    while (fgetc(file) != EOF) {
        characters++;
    }

    fclose(file);
    printf("Number of characters: %d\n", characters);
    exit(0);
}

int main() {
    pid_t c1, c2, c3;

    if ((c1 = fork()) == 0) {
        count_words();
    }

    if ((c2 = fork()) == 0) {
        count_lines();
    }

    if ((c3 = fork()) == 0) {
        count_characters();
    }

    waitpid(c1, NULL, 0);
    waitpid(c2, NULL, 0);
    waitpid(c3, NULL, 0);

    printf("All child processes completed.\n");
    return 0;
}
