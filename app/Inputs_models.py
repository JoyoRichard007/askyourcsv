# from langchain_core.pydantic_v1 import BaseModel, Field, validator
from pydantic import BaseModel, Field, validator
from typing import List
import re

class Message(BaseModel):
    role: str = Field(default="human")
    content: str | None

class Messages(BaseModel):
    messages: List[Message]

class PythonInputs(BaseModel):
    query: str = Field(description="Code snippet to run")

    @validator('query')
    def validate_query(cls, v):
        # Block imports
        if re.search(r'\bimport\b', v):
            return "Imports are not allowed"

        # Block dangerous functions and patterns
        dangerous_patterns = [
            r'\bos\.', r'\bsys\.', r'\bsubprocess\.', r'\bshutil\.', 
            r'\bexec\b', r'\beval\b', r'\bopen\b', r'\bsystem\b', 
            r'\bcompile\b', r'\binput\b', r'\bbuiltins\b',
            r'\bthreading\b', r'\bconcurrent\b', r'\basyncio\b',
            r'\bgetattr\b', r'\bsetattr\b', r'\bdelattr\b', r'\bos\.',
            r'\brun\b', r'\bPopen\b', r'\bpop\b', r'\bwhile\b', 
            r'\bfor\b', r'\btry\b', r'\bexcept\b', r'\bfinally\b', 
            r'\bwith\b', r'\bfrom\b', r'\bimport\b', r'\brmdir\b',
            r'\bunlink\b', r'\bmkdir\b', r'\bkill\b', r'\bchmod\b', 
            r'\bchown\b', r'\bchgrp\b', r'\brmtree\b', r'\bfork\b',
            r'\bspawn\b', r'\bpopen\b', r'\bsignal\b', r'\bresource\b',
            r'\binspect\b', r'\bglob\b', r'\bfnmatch\b', r'\bsocket\b',
            r'\burllib\b', r'\bhttp\b', r'\bftplib\b', r'\btelnetlib\b',
            r'\bsmtplib\b', r'\bssl\b', r'\bhashlib\b', r'\btempfile\b',
            r'\bos.chdir\b', r'\bos.getcwd\b', r'\bos.rename\b', 
            r'\bos.path\b', r'\bos.scandir\b', r'\bos.walk\b', 
            r'\bos.access\b', r'\bos.stat\b', r'\bos.getlogin\b',
            r'\bos.getuid\b', r'\bos.getgid\b', r'\bos.geteuid\b',
            r'\bos.getegid\b', r'\bos.getgroups\b', r'\bos.umask\b',
            r'\bos.setuid\b', r'\bos.setgid\b', r'\bos.seteuid\b',
            r'\bos.setegid\b', r'\bos.setgroups\b'
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, v):
                return "Dangerous functions are not allowed"

        # Block dataframe modification
        if re.search(r'\bdf\.\w+\s*=', v):
            return "Modifying the dataframe is not allowed"

        # Ensure code is safe for both cloud and local environments
        unsafe_keywords = [
            'importlib', 'pickle', 'marshal', 'sys.modules', 
            'sys.path', 'os.environ', 'os.system', 'os.popen', 
            'os.fork', 'os.spawn', 'os.exec', 'signal', 'resource', 
            'inspect', 'glob', 'fnmatch', 'socket', 'urllib', 'http',
            'ftplib', 'telnetlib', 'smtplib', 'ssl', 'hashlib', 'tempfile',
            'platform', 'ctypes', 'ctypes.util', 'multiprocessing', 
            'xml.etree', 'xml.dom', 'xml.sax', 'xml.parsers', 
            'csv', 'pdb', 'random.seed', 'random.getstate', 
            'random.setstate', 'numpy.load', 'numpy.save', 'numpy.savetxt'
        ]
        if any(kw in v for kw in unsafe_keywords):
            return "Unsafe keywords are not allowed"

        # Block operations that might affect the environment or execution flow
        environment_patterns = [
            r'\bos.chdir\b', r'\bos.getcwd\b', r'\bos.rename\b', 
            r'\bos.path\b', r'\bos.scandir\b', r'\bos.walk\b', 
            r'\bos.access\b', r'\bos.stat\b', r'\bos.getlogin\b',
            r'\bos.getuid\b', r'\bos.getgid\b', r'\bos.geteuid\b',
            r'\bos.getegid\b', r'\bos.getgroups\b', r'\bos.umask\b',
            r'\bos.setuid\b', r'\bos.setgid\b', r'\bos.seteuid\b',
            r'\bos.setegid\b', r'\bos.setgroups\b', r'\bos.getpid\b',
            r'\bos.getppid\b', r'\bos.setsid\b', r'\bos.setpgrp\b',
            r'\bos.tcgetpgrp\b', r'\bos.tcsetpgrp\b', r'\bos.getloadavg\b',
            r'\bos.urandom\b', r'\bos.symlink\b', r'\bos.readlink\b',
            r'\bos.link\b', r'\bos.fork\b', r'\bos.kill\b', r'\bos.abort\b',
            r'\bos.system\b', r'\bos.wait\b', r'\bos.wait3\b', r'\bos.wait4\b'
        ]
        for pattern in environment_patterns:
            if re.search(pattern, v):
                return "Environment or execution flow modifications are not allowed"

        # Final safety check to ensure no suspicious patterns are present
        suspicious_patterns = [
            r'\bcodecs\b', r'\bmmap\b', r'\btrace\b', r'\bweakref\b', 
            r'\bmultiprocessing\b', r'\bthread\b', r'\bsignal\b', 
            r'\basync\b', r'\bawait\b', r'\byield\b', r'\bmagic\b', 
            r'\bhook\b', r'\bcontextlib\b', r'\btempfile\b', r'\brunpy\b',
            r'\btarfile\b', r'\bzipfile\b', r'\bgzip\b', r'\bopen\b',
            r'\bshelve\b', r'\bmarshal\b', r'\bsignal\b', r'\bcProfile\b',
            r'\bpickle\b', r'\bdill\b', r'\bpyclbr\b', r'\bpy_compile\b',
            r'\bpydoc\b', r'\bsubprocess\b', r'\bos.popen\b', r'\bmagic\b'
        ]
        for pattern in suspicious_patterns:
            if re.search(pattern, v):
                return "Suspicious patterns are not allowed"

        # If the code passes all checks, validate it
        return v