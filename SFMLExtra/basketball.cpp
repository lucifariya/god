#include <SFML/Graphics.hpp>
#include <iostream>
#include <sstream>
using namespace sf;
using namespace std;

int main()
{
    VideoMode video(1980, 1020);
    RenderWindow window(video, "CatchBallGame");
    float windowWidth = 1980;
    float windowHeight = 1020;
    View view(FloatRect(0, 0, windowWidth, windowHeight));
    window.setView(view);

    // Basket (bat) setup
    RectangleShape batShape;
    float batWidth = 200;
    float batHeight = 10;
    float batSpeed = 2;
    float batPixSec = windowHeight / batSpeed;
    batShape.setOrigin(batWidth / 2, batHeight / 2);
    batShape.setSize(Vector2f(batWidth, batHeight));
    batShape.setFillColor(Color::White);
    batShape.setPosition(windowWidth / 2, windowHeight - 20);

    // White Ball (safe)
    CircleShape ballShape;
    float radius = 20;
    float ballSpeed = 3;
    float ballPixSecY = windowHeight / ballSpeed;
    ballShape.setRadius(radius);
    ballShape.setFillColor(Color::White);
    ballShape.setPosition(20 + rand() % (int)(windowWidth - 40), -100);

    // Red Balls (unsafe)
    const int numRedBalls = 6;
    CircleShape ballEnemyShape[numRedBalls];
    for (int i = 0; i < numRedBalls; i++)
    {
        ballEnemyShape[i].setRadius(radius);
        ballEnemyShape[i].setFillColor(Color::Red);
        float x = 20 + rand() % (int)(windowWidth - 40);
        float y = -200 + rand() % 200;
        ballEnemyShape[i].setPosition(x, y);
    }

    // HUD
    Font font;
    if (!font.loadFromFile("KOMIKAP_.ttf"))
    {
        cout << "Font not found!" << endl;
        return 1;
    }
    Text scoreText, timeText, messageText;
    scoreText.setFont(font);
    scoreText.setCharacterSize(75);
    scoreText.setFillColor(Color::White);
    scoreText.setPosition(20, 20);

    timeText.setFont(font);
    timeText.setCharacterSize(75);
    timeText.setFillColor(Color::White);

    messageText.setFont(font);
    messageText.setCharacterSize(100);
    messageText.setFillColor(Color::Yellow);
    messageText.setPosition(windowWidth / 2, windowHeight / 2);

    // Time bar
    float timeLimit = 20.0f; // seconds
    float timeLeft = timeLimit;
    RectangleShape timeBar;
    timeBar.setSize(Vector2f(windowWidth - 40, 30));
    timeBar.setFillColor(Color::Green);
    timeBar.setPosition(20, windowHeight - 60);

    // Game state
    int scoreVal = 0;
    bool paused = false;
    bool gameOver = false;
    Clock ct;
    Time dt;

    while (window.isOpen())
    {
        dt = ct.restart();
        Event ev;
        while (window.pollEvent(ev))
        {
            if (ev.type == Event::Closed || (ev.type == Event::KeyPressed && ev.key.code == Keyboard::Escape))
                window.close();
            if (ev.type == Event::KeyPressed && ev.key.code == Keyboard::Space && !gameOver)
            {
                paused = !paused;
            }
            if (ev.type == Event::KeyPressed && ev.key.code == Keyboard::Enter && gameOver)
            {
                // Restart game
                paused = false;
                gameOver = false;
                scoreVal = 0;
                timeLeft = timeLimit;
                ballShape.setPosition(windowWidth / 2, 20);
                for (int i = 0; i < numRedBalls; i++)
                {
                    float x = 20 + rand() % (int)(windowWidth - 40);
                    float y = -200 + rand() % 200;
                    ballEnemyShape[i].setPosition(x, y);
                }
            }
        }

        // HUD update
        stringstream ss, ssTime;
        ss << "Score: " << scoreVal;
        scoreText.setString(ss.str());

        if (!gameOver)
            ssTime << "Time: " << (int)timeLeft;
        else
            ssTime << "Game Over! Press Enter to restart";
        timeText.setString(ssTime.str());
        FloatRect details = timeText.getLocalBounds();
        timeText.setPosition(windowWidth - details.width - 20, 20);

        // Time bar update
        timeBar.setSize(Vector2f((windowWidth - 40) * (timeLeft / timeLimit), 30));
        if (timeLeft / timeLimit < 0.3f)
            timeBar.setFillColor(Color::Red);
        else if (timeLeft / timeLimit < 0.6f)
            timeBar.setFillColor(Color::Yellow);
        else
            timeBar.setFillColor(Color::Green);

        if (!paused && !gameOver)
        {
            // Move basket
            if (Keyboard::isKeyPressed(Keyboard::Left))
            {
                float x = batShape.getPosition().x;
                float y = batShape.getPosition().y;
                x -= dt.asSeconds() * batPixSec;
                if (x < batWidth / 2)
                    x = batWidth / 2;
                batShape.setPosition(x, y);
            }
            if (Keyboard::isKeyPressed(Keyboard::Right))
            {
                float x = batShape.getPosition().x;
                float y = batShape.getPosition().y;
                x += dt.asSeconds() * batPixSec;
                if (x > windowWidth - batWidth / 2)
                    x = windowWidth - batWidth / 2;
                batShape.setPosition(x, y);
            }

            // Move white ball
            float x = ballShape.getPosition().x;
            float y = ballShape.getPosition().y;
            y += ballPixSecY * dt.asSeconds();
            if (y > view.getSize().y)
            {
                y = -40;
                x = 20 + rand() % (int)(view.getSize().x - 40);
            }
            ballShape.setPosition(x, y);

            // Move red balls
            for (int i = 0; i < numRedBalls; i++)
            {
                float ex = ballEnemyShape[i].getPosition().x;
                float ey = ballEnemyShape[i].getPosition().y;
                ey += ballPixSecY * dt.asSeconds();
                if (ey > view.getSize().y)
                {
                    ey = -40;
                    ex = 20 + rand() % (int)(view.getSize().x - 40);
                }
                ballEnemyShape[i].setPosition(ex, ey);
            }

            // Collision: white ball
            if (ballShape.getGlobalBounds().intersects(batShape.getGlobalBounds()))
            {
                ballShape.setPosition(20 + rand() % (int)(view.getSize().x - 40), -40);
                scoreVal++;
                timeLeft += 2.0f; // reward time
                if (timeLeft > timeLimit)
                    timeLeft = timeLimit;
            }

            // Collision: red balls
            for (int i = 0; i < numRedBalls; i++)
            {
                if (ballEnemyShape[i].getGlobalBounds().intersects(batShape.getGlobalBounds()))
                {
                    ballEnemyShape[i].setPosition(20 + rand() % (int)(view.getSize().x - 40), -40);
                    timeLeft -= 5.0f; // penalty
                }
            }

            // Time update
            timeLeft -= dt.asSeconds();
            if (timeLeft <= 0)
            {
                timeLeft = 0;
                paused = true;
                gameOver = true;
            }
        }

        window.clear();
        window.draw(batShape);
        window.draw(ballShape);
        for (int i = 0; i < numRedBalls; i++)
        {
            window.draw(ballEnemyShape[i]);
        }
        window.draw(scoreText);
        window.draw(timeText);
        window.draw(timeBar);
        window.display();
    }
    return 0;
}