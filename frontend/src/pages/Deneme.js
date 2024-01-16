import React, { useEffect, useState } from 'react'
import {Modal} from "bootstrap"
const Deneme = () => {

    const [isModalOpen, setIsModalOpen] = useState(false);

    const openModal = () => {
        setIsModalOpen(true);
    }

    useEffect(() => {
        const intervalId = setInterval(() => {
            openModal();
        }, 1 * 10000);
        // Clean up the interval when the component unmounts
        return () => clearInterval(intervalId);
    }, []); // Empty dependency array to run the effect only once on mount

    useEffect(()=>{
        var myModal = new Modal(document.getElementById('exampleModal'))
        if(isModalOpen)
        {
            myModal.show()
        }else if (!isModalOpen){
            myModal.hide()
        }
        
    },[isModalOpen])

    return (
        
        <div class="modal fade modal-lg" id="exampleModal" tabindex="-1" role="dialog" aria-labelledby="exampleModalLabel" aria-hidden="true">
          <div class="modal-dialog" role="document">
            <div class="modal-content">
              <div class="modal-header">
                <h4 class="modal-title" id="exampleModalLabel">Kalp krizi riski</h4>
                <button type="button" class="close" data-bs-dismiss="modal" aria-label="Close">
                  <span aria-hidden="true">&times;</span>
                </button>
              </div>
              <div class="modal-body d-flex align-items-center justify-content-center">
                {
                  // loading ?
                  //   (
                  //     <div className='d-flex flex-column align-items-center justify-content-center gap-4'>
                  //       <FaHeartbeat color='red' size={48} />
                  //       <p className='h3'>Kalp krizi riski ölçülüyor</p>
                  //       <ReactLoading type='spin' color='black' width={'32px'} height={'32px'} />
                  //     </div>
                  //   ) :
                  (
                    <div className='d-flex flex-column align-items-center justify-content-center gap-2'>
                      Modal
                    </div>
                  )
                }
              </div>
              <div class="modal-footer">
                <button onClick={()=> {setIsModalOpen(false)}} type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
              </div>
            </div>
          </div>
        </div>
    )
}

export default Deneme