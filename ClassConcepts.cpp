#include <iostream>
#include <string>
using namespace std;

// Base class
class Vehicle
{
protected:
    string brand;
    int year;

public:
    Vehicle(string b, int y) : brand(b), year(y) {}
    virtual void display() const
    { // virtual function
        cout << "Brand: " << brand << ", Year: " << year << endl;
    }
    virtual ~Vehicle() {} // virtual destructor
};

// Derived class with inheritance and polymorphism
class Car : public Vehicle
{
    string model;
    static int carCount; // static member
public:
    Car(string b, int y, string m) : Vehicle(b, y), model(m) { carCount++; }
    void display() const override
    { // override virtual function
        cout << "Car - Brand: " << brand << ", Model: " << model << ", Year: " << year << endl;
    }
    static int getCarCount() { return carCount; } // static member function
    friend void showCarSecret(const Car &c);      // friend function
};
int Car::carCount = 0;

// Multiple inheritance
class Electric
{
protected:
    int batteryLife;

public:
    Electric(int bl) : batteryLife(bl) {}
    void showBattery() const
    {
        cout << "Battery Life: " << batteryLife << " km" << endl;
    }
};

// Derived class with multiple inheritance
class ElectricCar : public Car, public Electric
{
public:
    ElectricCar(string b, int y, string m, int bl)
        : Car(b, y, m), Electric(bl) {}
    void display() const override
    {
        Car::display();
        cout << " (Electric) ";
        showBattery();
    }
};

// Friend function definition
void showCarSecret(const Car &c)
{
    cout << "[Friend] Accessing Car's brand: " << c.brand << endl;
}

// Abstract class (interface)
class Printable
{
public:
    virtual void print() const = 0; // pure virtual
};

// Class with operator overloading and Printable interface
class Owner : public Printable
{
    string name;
    int age;

public:
    Owner(string n, int a) : name(n), age(a) {}
    void print() const override
    {
        cout << "Owner: " << name << ", Age: " << age << endl;
    }
    // Operator overloading
    bool operator==(const Owner &other) const
    {
        return name == other.name && age == other.age;
    }
};

int main()
{
    Car c1("Toyota", 2020, "Corolla");
    ElectricCar ec1("Tesla", 2022, "Model 3", 500);

    Vehicle *v1 = &c1;
    Vehicle *v2 = &ec1;

    v1->display(); // Polymorphism
    v2->display(); // Polymorphism

    showCarSecret(c1); // Friend function

    cout << "Total Cars: " << Car::getCarCount() << endl;

    Owner o1("Alice", 30), o2("Bob", 25), o3("Alice", 30);
    o1.print();
    o2.print();
    cout << "o1 == o3? " << (o1 == o3 ? "Yes" : "No") << endl;

    return 0;
}