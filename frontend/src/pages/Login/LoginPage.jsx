import React, { useState } from 'react'
import "./login.css"
import axios from "axios"
import { useNavigate } from "react-router-dom";
import { useAuth } from '../../contexts/AuthContext';

// Statik veriler=>yaş,cinsiyet,sigara kullanıp kullanmaması,günlük içilen sigara miktarı,kan basıncı için ilaç kullanıp kullanmaması,felç geçirip gecirmemesi,
// hipertansiyon hastası olup olmaması,diyabet hastası olup olmaması, vücut kitle indeksi
const LoginPage = () => {
  const [userInfo, setUserInfo] = useState({
    tcNo: "",
    sifre: "",
  })
  const [errorLogin,setErrorLogin] = useState(false)
  const {
        currentUser,
        setCurrentUser,
        isLoggedIn,
        setIsLoggedIn
  } = useAuth()

  const handleChange = (e) => {
    setUserInfo({ ...userInfo, [e.target.name]: e.target.value })
  }
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault()
    try{
      const res = await axios.post("http://localhost:4000/login", userInfo)
      console.log(res.data)
      setIsLoggedIn(true)
      setCurrentUser(res.data[0])
      localStorage.setItem("user",JSON.stringify(res.data[0]))
      navigate("/")
    }catch(err){
      setErrorLogin(true)
      console.log(err.message)
    }
  }
  console.log(userInfo)
  return (
    <div className='login-page d-flex align-items-center justify-content-center'>
      {/* {
        stepIndex == 0 ?
        (<StepOne setStepIndex={setStepIndex} stepIndex={stepIndex}/>):
        (<StepTwo setStepIndex={setStepIndex} stepIndex={stepIndex}/>)
      } */}
      <form action="">
        <h3 className='text-center'>Giriş</h3>
        <div className='login-form'>
          <div className='d-flex flex-column gap-1'>
            <label htmlFor=""><span style={{ color: "red" }}>*</span> Tc Kimlik No</label>
            <input required type="text" name='tcNo' onChange={handleChange} />
          </div>
          <div className='d-flex flex-column gap-1'>
            <label htmlFor=""><span style={{ color: "red" }}>*</span> Şifre</label>
            <input required type="password" name='sifre' onChange={handleChange} />
          </div>
          <div className='btn-div'>
            <div className='register-div' onClick={()=>navigate("/register")}>Kaydınız yoksa buraya tıklayın</div>
            <button onClick={handleLogin} className='btn btn-success'>Giriş Yap</button>
          </div>
          {
            errorLogin && <div className='text-danger'>Kullanıcı veya şifre bilginiz yanlış !</div>
          }
        </div>
      </form>
    </div>

  )
}

export default LoginPage