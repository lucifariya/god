
#include <stdio.h>

void countDenominations(int amount) {
    int denominations[] = {2000, 500, 200, 100, 50, 20, 10, 5, 2, 1};
    int count[10] = {0};
    int totalNotes = 0;

    for (int i = 0; i < 10; i++) {
        count[i] = amount / denominations[i];
        amount %= denominations[i];
        totalNotes += count[i];
    }

    printf("Denomination\tCount\n");
    for (int i = 0; i < 10; i++) {
        if (count[i] != 0)
            printf("%d\t\t%d\n", denominations[i], count[i]);
    }
    printf("Total Notes: %d\n", totalNotes);
}

int main() {
    int amount = 3867; // Example amount
    countDenominations(amount);
    return 0;
}
