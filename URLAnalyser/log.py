from colorama import Fore



class Log:
    # Determines how verbose the log messages should be
    # 0 = Default, show only results and error messages
    # 1 = Developer, show previous and info messages
    # 2 = Training, show previous and training messages
    verboseness = 0

    @staticmethod
    def info(message):
        if Log.verboseness > 0:
            print(Fore.LIGHTCYAN_EX + "[INFO]: " + message)

    @staticmethod
    def success(message):
        if Log.verboseness > 0:
            print(Fore.GREEN + "[SUCCESS] " + message)

    @staticmethod
    def error(message):
        if Log.verboseness > 0:
            print(Fore.RED + "[ERROR]: " + message)
        exit(-1)
    
    @staticmethod
    def result(message):
        print(Fore.WHITE + message)
