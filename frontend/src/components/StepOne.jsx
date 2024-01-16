import React, { useState } from 'react'
import "../pages/Register/register.css"
const StepOne = ({setStepIndex,stepIndex}) => {
    const [userInfo,setUserInfo] = useState({
        tc:"",
        name:"",
        surname:"",
        password:"",
    })

    const handleChange = (e)=>{
        setUserInfo({...userInfo,[e.target.name]:e.target.value})
    }

    console.log(userInfo)
    return (
        <form action="" className='register-form d-flex flex-column gap-4'>
            <div className='d-flex flex-column gap-1'>
                <label htmlFor="">Tc Kimlik No</label>
                <input type="text" name='tc' onChange={(e)=>handleChange(e)}/>
            </div>
            <div className='d-flex flex-column gap-1'>
                <label htmlFor="">Ad</label>
                <input type="text" name='name' onChange={(e)=>handleChange(e)}/>
            </div>
            <div className='d-flex flex-column gap-1'>
                <label htmlFor="">Soyad</label>
                <input type="text" name='surname' onChange={(e)=>handleChange(e)}/>
            </div>
            <div className='d-flex flex-column gap-1'>
                <label htmlFor="">Şifre</label>
                <input type="password" name='password' onChange={(e)=>handleChange(e)}/>
            </div>
            <button onClick={()=>setStepIndex(stepIndex + 1)} className='btn btn-success'>Next</button>
        </form>
    )
}

export default StepOne