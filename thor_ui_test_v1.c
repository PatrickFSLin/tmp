#include <X11/Xlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>

#define WIDTH  1280
#define HEIGHT 720

#define BEV_X 640
#define BEV_Y 80
#define BEV_W 580
#define BEV_H 540

/* ---------------------------------------------------------
 * Utility
 * --------------------------------------------------------- */

void draw_string(
    Display *display,
    Window window,
    GC gc,
    int x,
    int y,
    const char *text)
{
    XDrawString(display, window, gc, x, y, text, strlen(text));
}


/* ---------------------------------------------------------
 * Coordinate transformation
 *
 * World coordinate:
 *
 *       +Y
 *        ^
 *        |
 *        |
 *        O ----> +X
 *
 * BEV:
 *
 *       front
 *        ^
 *        |
 *       ego
 *
 * --------------------------------------------------------- */

int world_to_bev_x(double x)
{
    double scale = 35.0;

    return BEV_X + BEV_W / 2 + (int)(x * scale);
}

int world_to_bev_y(double y)
{
    double scale = 35.0;

    /*
     * Positive Y = forward
     * Therefore screen Y decreases.
     */

    return BEV_Y + BEV_H - 80 - (int)(y * scale);
}


/* ---------------------------------------------------------
 * Draw BEV grid
 * --------------------------------------------------------- */

void draw_bev_grid(
    Display *display,
    Window window,
    GC gc)
{
    int center_x = BEV_X + BEV_W / 2;
    int bottom_y = BEV_Y + BEV_H - 80;

    /* BEV border */

    XDrawRectangle(
        display,
        window,
        gc,
        BEV_X,
        BEV_Y,
        BEV_W,
        BEV_H
    );

    /* Vertical center line */

    XDrawLine(
        display,
        window,
        gc,
        center_x,
        BEV_Y,
        center_x,
        BEV_Y + BEV_H
    );

    /* Horizontal reference lines */

    for (int i = 1; i <= 5; i++)
    {
        int y = bottom_y - i * 70;

        XDrawLine(
            display,
            window,
            gc,
            BEV_X,
            y,
            BEV_X + BEV_W,
            y
        );
    }

    /* Lane lines */

    int lane_left  = center_x - 100;
    int lane_right = center_x + 100;

    XDrawLine(
        display,
        window,
        gc,
        lane_left,
        BEV_Y,
        lane_left,
        BEV_Y + BEV_H
    );

    XDrawLine(
        display,
        window,
        gc,
        lane_right,
        BEV_Y,
        lane_right,
        BEV_Y + BEV_H
    );
}


/* ---------------------------------------------------------
 * Draw ego vehicle
 * --------------------------------------------------------- */

void draw_ego_vehicle(
    Display *display,
    Window window,
    GC gc)
{
    int center_x = BEV_X + BEV_W / 2;
    int bottom_y = BEV_Y + BEV_H - 80;

    int car_w = 50;
    int car_h = 90;

    XFillRectangle(
        display,
        window,
        gc,
        center_x - car_w / 2,
        bottom_y - car_h,
        car_w,
        car_h
    );

    /* Front direction marker */

    XDrawLine(
        display,
        window,
        gc,
        center_x,
        bottom_y - car_h,
        center_x,
        bottom_y - car_h - 20
    );
}


/* ---------------------------------------------------------
 * Draw fake surrounding vehicles
 * --------------------------------------------------------- */

void draw_other_vehicle(
    Display *display,
    Window window,
    GC gc,
    double x,
    double y)
{
    int px = world_to_bev_x(x);
    int py = world_to_bev_y(y);

    int w = 35;
    int h = 60;

    XDrawRectangle(
        display,
        window,
        gc,
        px - w / 2,
        py - h / 2,
        w,
        h
    );
}


/* ---------------------------------------------------------
 * Draw fake trajectory
 *
 * Curved trajectory changes with frame number.
 * --------------------------------------------------------- */

void draw_fake_trajectory(
    Display *display,
    Window window,
    GC gc,
    int frame)
{
    const int points = 40;

    int previous_x = 0;
    int previous_y = 0;

    for (int i = 0; i < points; i++)
    {
        double y = i * 0.30;

        /*
         * Slowly changing curvature.
         */

        double curvature =
            0.8 * sin(frame * 0.025);

        double x =
            curvature * (y * y) / 5.0;

        int px = world_to_bev_x(x);
        int py = world_to_bev_y(y);

        if (i > 0)
        {
            XDrawLine(
                display,
                window,
                gc,
                previous_x,
                previous_y,
                px,
                py
            );
        }

        previous_x = px;
        previous_y = py;
    }
}


/* ---------------------------------------------------------
 * Main
 * --------------------------------------------------------- */

int main(void)
{
    Display *display;
    Window window;
    GC gc;
    XEvent event;

    display = XOpenDisplay(NULL);

    if (!display)
    {
        fprintf(stderr, "ERROR: Cannot open X display\n");
        return 1;
    }

    int screen = DefaultScreen(display);

    window = XCreateSimpleWindow(
        display,
        RootWindow(display, screen),
        100,
        50,
        WIDTH,
        HEIGHT,
        1,
        WhitePixel(display, screen),
        BlackPixel(display, screen)
    );

    XStoreName(
        display,
        window,
        "VaVAM on DRIVE AGX Thor - BEV Demo"
    );

    XSelectInput(
        display,
        window,
        ExposureMask | KeyPressMask
    );

    XMapWindow(display, window);

    gc = XCreateGC(display, window, 0, NULL);

    int frame = 0;

    struct timespec start_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    while (1)
    {
        /* -------------------------------------------------
         * Handle keyboard
         * ------------------------------------------------- */

        while (XPending(display))
        {
            XNextEvent(display, &event);

            if (event.type == KeyPress)
            {
                XFreeGC(display, gc);
                XDestroyWindow(display, window);
                XCloseDisplay(display);

                return 0;
            }
        }

        /* -------------------------------------------------
         * Clear screen
         * ------------------------------------------------- */

        XSetForeground(
            display,
            gc,
            BlackPixel(display, screen)
        );

        XFillRectangle(
            display,
            window,
            gc,
            0,
            0,
            WIDTH,
            HEIGHT
        );

        /* -------------------------------------------------
         * UI text
         * ------------------------------------------------- */

        XSetForeground(
            display,
            gc,
            WhitePixel(display, screen)
        );

        XDrawString(
            display,
            window,
            gc,
            35,
            40,
            "VaVAM on DRIVE AGX Thor - BEV Demo",
            35
        );

        /* -------------------------------------------------
         * Camera placeholder
         * ------------------------------------------------- */

        XDrawRectangle(
            display,
            window,
            gc,
            35,
            80,
            540,
            540
        );

        XDrawString(
            display,
            window,
            gc,
            230,
            330,
            "CAMERA PLACEHOLDER",
            18
        );

        XDrawString(
            display,
            window,
            gc,
            245,
            360,
            "VaVAM Open-Loop",
            16
        );

        /* -------------------------------------------------
         * BEV title
         * ------------------------------------------------- */

        XDrawString(
            display,
            window,
            gc,
            BEV_X,
            60,
            "BEV / PREDICTED TRAJECTORY",
            27
        );

        /* -------------------------------------------------
         * BEV
         * ------------------------------------------------- */

        draw_bev_grid(
            display,
            window,
            gc
        );

        /* -------------------------------------------------
         * Ego
         * ------------------------------------------------- */

        draw_ego_vehicle(
            display,
            window,
            gc
        );

        /* -------------------------------------------------
         * Fake surrounding vehicles
         * ------------------------------------------------- */

        draw_other_vehicle(
            display,
            window,
            gc,
            -1.8,
            5.0
        );

        draw_other_vehicle(
            display,
            window,
            gc,
            2.0,
            8.0
        );

        draw_other_vehicle(
            display,
            window,
            gc,
            -2.5,
            10.5
        );

        /* -------------------------------------------------
         * Fake trajectory
         * ------------------------------------------------- */

        draw_fake_trajectory(
            display,
            window,
            gc,
            frame
        );

        /* -------------------------------------------------
         * Bottom information
         * ------------------------------------------------- */

        char text[256];

        snprintf(
            text,
            sizeof(text),
            "Frame: %06d",
            frame
        );

        XDrawString(
            display,
            window,
            gc,
            35,
            665,
            text,
            strlen(text)
        );

        snprintf(
            text,
            sizeof(text),
            "FPS: ~60"
        );

        XDrawString(
            display,
            window,
            gc,
            230,
            665,
            text,
            strlen(text)
        );

        XDrawString(
            display,
            window,
            gc,
            360,
            665,
            "Inference: FAKE",
            15
        );

        XDrawString(
            display,
            window,
            gc,
            570,
            665,
            "Trajectory: FAKE",
            17
        );

        XDrawString(
            display,
            window,
            gc,
            810,
            665,
            "Press any key to exit",
            21
        );

        XFlush(display);

        frame++;

        /*
         * ~60 FPS
         */

        usleep(16666);
    }

    return 0;
}
