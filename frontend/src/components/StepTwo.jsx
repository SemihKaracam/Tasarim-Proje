import React from 'react'
import "../pages/Register/register.css"

const StepTwo = ({ setStepIndex, stepIndex }) => {
    return (
            <form action="" className='register-form d-flex flex-column gap-4'>
                <div className='d-flex flex-column gap-1'>
                    <label htmlFor="">Cinsiyet</label>
                    <select>
                        <option>Erkek</option>
                        <option>Kadın</option>
                    </select>
                </div>
                <div className='d-flex flex-column gap-1'>
                    <label htmlFor="">Yaş</label>
                    <input type="number" min="1" max="150" />
                </div>
                <div className='d-flex flex-column gap-1'>
                    <label htmlFor="">Sigara kullanıyor musunuz ?</label>
                    <select>
                        <option>Evet</option>
                        <option>Hayır</option>
                    </select>        
                </div>
                <div className='d-flex flex-column gap-1'>
                    <label htmlFor="">Kan basıncı için ilaç kullanıyor musunuz ?</label>""
                    <select>
                        <option>Evet</option>
                        <option>Hayır</option>
                    </select>
                </div>
                <div className='d-flex flex-column gap-1'>
                    <label htmlFor="">Geçmişte felç geçirdiniz mi ?</label>
                    <select>
                        <option>Evet</option>
                        <option>Hayır</option>
                    </select>
                </div>
                <div className='d-flex flex-column gap-1'>
                    <label htmlFor="">Hipertansiyon hastalığınız var mı ?</label>
                    <select>
                        <option>Evet</option>
                        <option>Hayır</option>
                    </select>
                </div>
                <div className='d-flex flex-column gap-1'>
                    <label htmlFor="">Diyabet hastalığınız var mı ?</label>
                    <select>
                        <option>Evet</option>
                        <option>Hayır</option>
                    </select>
                </div>
                <div className='d-flex flex-column gap-1'>
                    <label htmlFor="">Vücut kitle endeksiniz</label>
                    <input type="text" />
                </div>
                <div className='d-flex align-item-center justify-content-between'>
                    <button onClick={() => setStepIndex(stepIndex - 1)}>Önceki</button>
                    <button className='btn btn-success'>Gönder</button>
                </div>
            </form>
    )
}

export default StepTwo