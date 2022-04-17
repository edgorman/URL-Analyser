from colorama import Fore


class Log:
    '''
        Handles all print statements to the terminal

        The 'verboseness' variable controls how verbose the log messages should be:
            * 0 = Default, show only results and error messages
            * 1 = Developer, include info messages
    '''
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
        print(Fore.RED + "[ERROR]: " + message)
        Log.result("Exiting due to an error within the module. For more info re-run the program with the verbose flag.")
        exit(-1)

    @staticmethod
    def result(message):
        print(Fore.WHITE + message)
