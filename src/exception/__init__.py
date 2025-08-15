import os 
import sys
import types

class CustomException(Exception):
    def __init__(self, error_message: Exception, error_detailes: types.ModuleType):
        self.error_message = CustomException.get_detailed_error_message(error_message=error_message,
                                                                        error_detailes=error_detailes)
        
#try ->
#exception ->
# a,b,c = 1,2,3
# _, _, c
    @staticmethod
    def get_detailed_error_message(error_message: Exception, error_detailes: types.ModuleType):
        _, _, exce_tb = error_detailes.exc_info()

        if exce_tb is None:
            return "No traceback available"

        exception_block_line_number = exce_tb.tb_frame.f_lineno
        try_block_line_number = exce_tb.tb_lineno
        file_name = exce_tb.tb_frame.f_code.co_filename

        error_message = f"""
        Error occured in execution of :
        [{file_name}] at
        try block line number : [{try_block_line_number}]
        and exception block line number : [{exception_block_line_number}]
        error message : [{error_message}]"""
        
        return error_message
    
    def __str__(self):
        return self.error_message
    
    def __repr__(self):
        return f"{CustomException.__name__}({self.error_message!r})"
