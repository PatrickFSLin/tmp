#include <X11/Xlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

int main(void)
{
    Display *display;
    Window window;
    GC gc;
    XEvent event;

    int screen;
    int width = 1280;
    int height = 720;

    display = XOpenDisplay(NULL);

    if (!display) {
        fprintf(stderr, "ERROR: Cannot open X display\n");
        return 1;
    }

    screen = DefaultScreen(display);

    window = XCreateSimpleWindow(
        display,
        RootWindow(display, screen),
        100, 100,
        width, height,
        1,
        BlackPixel(display, screen),
        WhitePixel(display, screen)
    );

    XStoreName(display, window, "VaVAM on DRIVE AGX Thor");

    XSelectInput(
        display,
        window,
        ExposureMask | KeyPressMask
    );

    XMapWindow(display, window);

    gc = XCreateGC(display, window, 0, NULL);

    int frame = 0;

    while (1)
    {
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

        /* Background */
        XSetForeground(
            display,
            gc,
            BlackPixel(display, screen)
        );

        XFillRectangle(
            display,
            window,
            gc,
            0, 0,
            width,
            height
        );

        /* Title */
        XSetForeground(
            display,
            gc,
            WhitePixel(display, screen)
        );

        XDrawString(
            display,
            window,
            gc,
            40,
            50,
            "VaVAM on DRIVE AGX Thor - UI Test",
            34
        );

        /* Test rectangle */
        XDrawRectangle(
            display,
            window,
            gc,
            450,
            200,
            380,
            180
        );

        XDrawString(
            display,
            window,
            gc,
            600,
            300,
            "TEST",
            4
        );

        /* Status */
        char status[128];

        snprintf(
            status,
            sizeof(status),
            "Frame: %06d",
            frame
        );

        XDrawString(
            display,
            window,
            gc,
            40,
            650,
            status,
            13
        );

        XDrawString(
            display,
            window,
            gc,
            40,
            680,
            "Status: RUNNING",
            14
        );

        XFlush(display);

        frame++;

        usleep(16666);  /* ~60 FPS */
    }

    return 0;
}
