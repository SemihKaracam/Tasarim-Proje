import React, { useState } from 'react'
import "./register.css"
import MultiStep from 'react-multistep'
import StepOne from '../../components/StepOne'
import StepTwo from '../../components/StepTwo'
import PrevButton from '../../components/PrevButton'
import NextButton from '../../components/NextButton'
import axios from "axios"
import { useNavigate } from "react-router-dom";

// Statik veriler=>yaş,cinsiyet,sigara kullanıp kullanmaması,günlük içilen sigara miktarı,kan basıncı için ilaç kullanıp kullanmaması,felç geçirip gecirmemesi,
// hipertansiyon hastası olup olmaması,diyabet hastası olup olmaması, vücut kitle indeksi
const RegisterPage = () => {
  const [userInfo, setUserInfo] = useState({
    tcNo: "",
    ad: "",
    soyad: "",
    sifre: "",
    cinsiyet: 1,
    yas: "",
    sigara: 0,
    kanbasinci: 0,
    felc: 0,
    hipertansiyon: 0,
    diyabet: 0,
    sigaraGunde: 0,
    vki: "",
    mail:""
  })

  const navigate = useNavigate();

  const handleChange = (e) => {
    setUserInfo({ ...userInfo, [e.target.name]: e.target.value })
  }

  const handleRegister = async (e) => {
    e.preventDefault()
    await axios.post("http://localhost:4000/register", userInfo)
    navigate("/login")
  }

  console.log(userInfo)
  return (
    <div className='register-page d-flex align-items-center justify-content-center'>
      {/* {
        stepIndex == 0 ?
        (<StepOne setStepIndex={setStepIndex} stepIndex={stepIndex}/>):
        (<StepTwo setStepIndex={setStepIndex} stepIndex={stepIndex}/>)
      } */}
      <form action="" onSubmit={handleRegister}>
        <h3 className='text-center' style={{fontSize:'32px'}}>Hasta Kayıt Sistemi</h3>
        <div className='register-form'>
          <div className='d-flex flex-column gap-1'>
            <label htmlFor=""><span style={{ color: "red" }}>*</span> Ad</label>
            <input required type="text" name='ad' onChange={handleChange} />
          </div>
          <div className='d-flex flex-column gap-1'>
            <label htmlFor=""><span style={{ color: "red" }}>*</span> Soyad</label>
            <input required type="text" name='soyad' onChange={handleChange} />
          </div>
          <div className='d-flex flex-column gap-1'>
            <label htmlFor=""><span style={{ color: "red" }}>*</span> Tc Kimlik No</label>
            <input minLength='11' maxLength='11' required type="text" name='tcNo' onChange={handleChange} />
          </div>
          <div className='d-flex flex-column gap-1'>
            <label htmlFor=""><span style={{ color: "red" }}>*</span>Email Adresi</label>
            <input required type="email" name='mail' onChange={handleChange} />
          </div>
          <div className='d-flex flex-column gap-1'>
            <label htmlFor=""><span style={{ color: "red" }}>*</span> Şifre</label>
            <input required type="password" name='sifre' onChange={handleChange} />
          </div>
          <div className='d-flex flex-column gap-1'>
            <label htmlFor=""><span style={{ color: "red" }}>*</span> Cinsiyet</label>
            <select defaultValue={1} name='cinsiyet' required onChange={handleChange}>
              <option value={1}>Erkek</option>
              <option value={0}>Kadın</option>
            </select>
          </div>
          <div className='d-flex flex-column gap-1'>
            <label htmlFor=""><span style={{ color: "red" }}>*</span> Yaş</label>
            <input name='yas' required onChange={handleChange} type="number" min="1" max="150" />
          </div>
          <div className='d-flex flex-column gap-1'>
            <label htmlFor=""><span style={{ color: "red" }}>*</span> Sigara kullanıyor musunuz ?</label>
            <select defaultValue={0} name='sigara' required onChange={handleChange}>
              <option value={1}>Evet</option>
              <option value={0}>Hayır</option>
            </select>
            {
              userInfo.sigara == 1 &&
              (
                <div className='d-flex flex-column gap-1'>
                  <label htmlFor=""><span style={{ color: "red" }}>*</span>Günde kaç adet sigara kullanıyorsunuz ? </label>
                  <input defaultValue={1} name='sigaraGunde' required onChange={handleChange}/>
                </div>
              )
            }
          </div>
          <div className='d-flex flex-column gap-1'>
            <label htmlFor=""><span style={{ color: "red" }}>*</span> Kan basıncı için ilaç kullanıyor musunuz ?</label>
            <select defaultValue={0} name='kanbasinci' required onChange={handleChange}>
              <option value={1}>Evet</option>
              <option value={0}>Hayır</option>
            </select>
          </div>
          <div className='d-flex flex-column gap-1'>
            <label htmlFor=""><span style={{ color: "red" }}>*</span> Geçmişte felç geçirdiniz mi ?</label>
            <select defaultValue={0} name='felc' required onChange={handleChange}>
              <option value={1}>Evet</option>
              <option value={0}>Hayır</option>
            </select>
          </div>
          <div className='d-flex flex-column gap-1'>
            <label htmlFor=""><span style={{ color: "red" }}>*</span> Hipertansiyon hastalığınız var mı ?</label>
            <select name='hipertansiyon' defaultValue={0} required onChange={handleChange}>
              <option value={1}>Evet</option>
              <option value={0}>Hayır</option>
            </select>
          </div>
          <div className='d-flex flex-column gap-1'>
            <label htmlFor=""><span style={{ color: "red" }}>*</span> Diyabet hastalığınız var mı ?</label>
            <select name='diyabet' defaultValue={0} required onChange={handleChange}>
              <option value={1}>Evet</option>
              <option value={0}>Hayır</option>
            </select>
          </div>
          <div className='d-flex flex-column gap-1'>
            <label htmlFor=""><span style={{ color: "red" }}>*</span> Vücut kitle endeksiniz</label>
            <input name='vki' required type="text" onChange={handleChange} />
          </div>
        </div>
        <div className='btn-div'>
          <button type='submit' className='btn btn-success'>Kayıt Ol</button>
        </div>
      </form>
    </div>

  )
}

export default RegisterPage