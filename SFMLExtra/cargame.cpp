#include <SFML/Graphics.hpp>
#include <iostream>
#include <sstream>
using namespace sf;
using namespace std;

int main()
{
    VideoMode video(200, 600);
    RenderWindow window(video, "CarGame", Style::Titlebar | Style::Close);

    RectangleShape divider[11];
    for (int i = 0; i < 11; i++)
    {
        divider[i].setSize(Vector2f(10, 50));
        divider[i].setPosition(95, -60 + (i * 60));
        divider[i].setFillColor(Color::White);
    }
    float dividerSpeed = 5;
    float diviserPixelPerSec = 600 / dividerSpeed;

    RectangleShape carDrive;
    carDrive.setSize(Vector2f(30, 50));
    carDrive.setFillColor(Color::White);
    carDrive.setPosition(40, 540);

    float carSpeed = 1;
    float carPixelPerSec = 200 / carSpeed;

    RectangleShape enemyCar[7];
    for (int i = 0; i < 7; i++)
    {
        enemyCar[i].setSize(Vector2f(30, 50));
        float x = rand() % 170;
        if (rand() % 3 != 0)
            enemyCar[i].setPosition(x, -900 + (i * 150));
        else
            enemyCar[i].setPosition(210, -900 + (i * 150));
        enemyCar[i].setFillColor(Color::Red);
    }
    float enemyCarPixelPerSec = 60;

    Clock ct;
    Time dt;
    bool gameOver = false;
    bool paused = false;
    bool acceptInput = true;
    int score = 0;
    int gameSpeed = 1;

    Text scoreHud;
    Text speedHud;
    Text message;
    Font ft;
    ft.loadFromFile("KOMIKAP_.ttf");
    scoreHud.setFont(ft);
    scoreHud.setFillColor(Color::White);
    scoreHud.setCharacterSize(12);
    scoreHud.setString("Score:0");
    scoreHud.setPosition(20, 20);

    speedHud.setFont(ft);
    speedHud.setFillColor(Color::White);
    speedHud.setCharacterSize(12);
    speedHud.setString("Speed:1");
    FloatRect speedHudBound = speedHud.getLocalBounds();
    speedHud.setPosition(window.getSize().x - 20 - speedHudBound.width, 20);

    message.setFont(ft);
    message.setFillColor(Color::White);
    message.setCharacterSize(20);
    message.setString("");
    FloatRect messageHudBound = message.getLocalBounds();
    message.setOrigin(messageHudBound.width / 2, messageHudBound.height / 2);
    message.setPosition(window.getSize().x / 2, window.getSize().y / 2);

    while (window.isOpen())
    {
        dt = ct.restart();
        Event event;
        while (window.pollEvent(event))
        {
            if (event.type == Event::Closed)
            {
                window.close();
            }
            if (event.type == Event::KeyReleased)
            {
                acceptInput = true;
            }
            if (event.type == Event::KeyPressed)
            {
                if (event.key.code == Keyboard::Enter && gameOver)
                {
                    paused = false;
                    gameOver = false;
                    score = 0;
                    enemyCarPixelPerSec = 60;
                    gameSpeed = 1;
                    for (int i = 0; i < 7; i++)
                    {
                        enemyCar[i].setSize(Vector2f(30, 50));
                        float x = rand() % 170;
                        if (rand() % 3 == 0)
                            enemyCar[i].setPosition(x, -900 + (i * 150));
                        else
                            enemyCar[i].setPosition(210, -900 + (i * 150));
                        enemyCar[i].setFillColor(Color::Red);
                    }
                    message.setString("");
                    messageHudBound = message.getLocalBounds();
                    message.setOrigin(messageHudBound.width / 2, messageHudBound.height / 2);
                    message.setPosition(window.getSize().x / 2, window.getSize().y / 2);
                    acceptInput = false;
                }
            }
        }
        if (Keyboard::isKeyPressed(Keyboard::Space) && acceptInput)
        {
            paused = !paused;
            acceptInput = false;
        }
        if (!paused)
        {
            if (Keyboard::isKeyPressed(Keyboard::Left))
            {
                float x = carDrive.getPosition().x;
                float y = carDrive.getPosition().y;
                x = x - dt.asSeconds() * carPixelPerSec;
                if (x < 0)
                    x = 0;
                carDrive.setPosition(x, y);
            }
            if (Keyboard::isKeyPressed(Keyboard::Right))
            {
                float x = carDrive.getPosition().x;
                float y = carDrive.getPosition().y;
                x = x + dt.asSeconds() * carPixelPerSec;
                if (x > 170)
                    x = 170;
                carDrive.setPosition(x, y);
            }
            stringstream ss, ss1;
            ss << "Score:" << score;
            scoreHud.setString(ss.str());
            ss1 << "Speed:" << gameSpeed;
            speedHud.setString(ss1.str());
            for (int i = 0; i < 7; i++)
            {
                float x = enemyCar[i].getPosition().x;
                float y = enemyCar[i].getPosition().y;
                y = y + enemyCarPixelPerSec * dt.asSeconds();
                if (y > 600)
                {
                    score = score + 1;
                    if (score % 10 == 0)
                    {
                        gameSpeed += 1;
                        enemyCarPixelPerSec = 60 + (gameSpeed * 10);
                    }
                    y = -450;
                    if (rand() % 3 != 0)
                        x = rand() % 170;
                    else
                        x = 210;
                }
                enemyCar[i].setPosition(x, y);
            }
            for (int i = 0; i < 11; i++)
            {
                float y = divider[i].getPosition().y;
                y = y + diviserPixelPerSec * dt.asSeconds();
                if (y > 600)
                {
                    y = -60;
                }
                divider[i].setPosition(95, y);
            }
            for (int i = 0; i < 7; i++)
            {
                if (enemyCar[i].getGlobalBounds().intersects(carDrive.getGlobalBounds()))
                {
                    gameOver = true;
                }
            }
        }
        if (gameOver)
        {
            paused = true;
            message.setString("GameOver!!!");
            messageHudBound = message.getLocalBounds();
            message.setOrigin(messageHudBound.width / 2, messageHudBound.height / 2);
            message.setPosition(window.getSize().x / 2, window.getSize().y / 2);
        }
        else if (paused && !gameOver)
        {
            message.setString("Paused!!!");
            messageHudBound = message.getLocalBounds();
            message.setOrigin(messageHudBound.width / 2, messageHudBound.height / 2);
            message.setPosition(window.getSize().x / 2, window.getSize().y / 2);
        }
        else
        {
            message.setString("");
            messageHudBound = message.getLocalBounds();
            message.setOrigin(messageHudBound.width / 2, messageHudBound.height / 2);
            message.setPosition(window.getSize().x / 2, window.getSize().y / 2);
        }
        window.clear();
        for (int i = 0; i < 11; i++)
        {
            window.draw(divider[i]);
        }
        for (int i = 0; i < 7; i++)
        {
            window.draw(enemyCar[i]);
        }
        window.draw(carDrive);
        window.draw(scoreHud);
        window.draw(speedHud);
        window.draw(message);
        window.display();
    }
    return 0;
}