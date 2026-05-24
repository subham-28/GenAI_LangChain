from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = "Subham" #default attribute
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5, description= "A decimal value representing the cgpa of the student")

new_student = {'name':'Subham', 'email':'abc@yahoo.com'}

student = Student(**new_student)

#dictionary
student_dict = dict(student)
print(student_dict['email'])

#json
student_json = student.model_dump_json()
