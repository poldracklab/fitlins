from .cli.run import main

if __name__ == "__main__":
    import sys

    from . import __name__

    # `python -m <module>` typically displays the command as __main__.py
    if "__main__.py" in sys.argv[0]:
        sys.argv[0] = "{} -m {}".format(sys.executable, __name__)
    main()
