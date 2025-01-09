#include <stdio.h>

int main() {
    int prev, curr, x;
    double c1 = 0, c2 = 0, c3 = 0, mnance = 50.0, sur = 0, totalc;

    printf("Enter the previous meter reading: ");
    scanf("%d", &prev);
    printf("Enter the current meter reading: ");
    scanf("%d", &curr);

    x = curr - prev;

    if (x > 300) {
        c3 = (x - 300) * 7.0;
        c2 = 200 * 5.0;
        c1 = 100 * 3.0;
    } else if (x > 100) {
        c2 = (x - 100) * 5.0;
        c1 = 100 * 3.0;
    } else {
        c1 = x * 3.0;
    }

    totalc = c1 + c2 + c3 + mnance;

    if (totalc > 1000) {
        sur = totalc * 0.10;
    }

    totalc += sur;

    printf("\nTotal units consumed: %d\n", x);
    printf("| %-10s | %-8s | %-8s | %-10s |\n", "Slab", "Rate", "Consumed", "Cost");
    printf("| %-10s | %-8s | %-8d | %-10.2f |\n", "1 (<100)", "@3.00", x > 100 ? 100 : x, c1);
    printf("| %-10s | %-8s | %-8d | %-10.2f |\n", "2 (101-300)", "@5.00", x > 300 ? 200 : (x > 100 ? x - 100 : 0), c2);
    printf("| %-10s | %-8s | %-8d | %-10.2f |\n", "3 (>300)", "@7.00", x > 300 ? x - 300 : 0, c3);
    printf("| %-10s | %-8s | %-8s | %-10.2f |\n", "Maintenance", "@50.00", "", mnance);
    if (sur > 0) {
        printf("| %-10s | %-8s | %-8s | %-10.2f |\n", "sur", "@10% (>1000)", "", sur);
    }
    printf("| %-10s | %-8s | %-8s | %-10.2f |\n", "Total Bill", "", "", totalc);

    return 0;
}
