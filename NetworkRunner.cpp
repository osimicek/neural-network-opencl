#include <iostream>
#include "NetworkRunner.h"

#define USE_CHRONO

#ifdef USE_CHRONO
#include <chrono>
#endif

/**
 * Prepares context and compiles kernel.
 */
NetworkRunner::NetworkRunner(uint platformId, uint deviceId, NetworkContainer *container):networks_container(container) {
    cl_int status;
    this->buf_taskdata = NULL;
    Device device = OpenclHelper::get_device(platformId, deviceId);
    this->max_neurons = 256;

    VECTOR_CLASS<Device> devices;
    devices.push_back(device);

    this->ctx = new Context(devices, NULL, NULL, NULL, &status);

    cl_command_queue_properties qprops = 0;
    qprops |= CL_QUEUE_PROFILING_ENABLE;
    qprops |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;

    this->queue = new CommandQueue(*this->ctx, device, qprops, &status);
    /**
     * Load kernel codes
     */
    char *knl_text = OpenclHelper::read_file("neural_network.cl");
    /**
     * Compile program
     */
    std::vector<std::string> codes;
    codes.push_back(knl_text);
    Program program = OpenclHelper::build_program(*this->ctx, codes, (std::string("-cl-mad-enable -cl-strict-aliasing -cl-denorms-are-zero -cl-single-precision-constant -DSHARED_MEMORY_SIZE=") + std::to_string(this->networks_container->get_shared_memory_per_network())).c_str());
    free(knl_text);
    /**
     * Get kernel
     */
    this->knl_learn = new Kernel(program, "run_neural_network", &status);
    this->knl_classification = new Kernel(program, "run_neural_network_classification", &status);
    CHECK_CL_ERROR(status, "cl::Kernel");
}

NetworkRunner::~NetworkRunner() {
    delete this->ctx;
    delete this->queue;
    delete this->knl_learn;
    delete this->knl_classification;
    if (this->buf_taskdata != NULL) {
        delete this->buf_taskdata;
        delete this->buf_task_data_transform;
    }
}

void NetworkRunner::set_max_neurons(int value) {
    this->max_neurons = value;
}

/**
 * Sends taskData to device.
 */
void NetworkRunner::write_task_data() {
    cl_int status;
    if (this->buf_taskdata != NULL) {
        delete this->buf_taskdata;
    }
    task_data_transform_t *task_data_transform = this->networks_container->get_task_data_transform();
    this->task_data_buffer = this->networks_container->get_task_data_buffer();
    this->task_data_buffer_size = this->networks_container->get_task_data_buffer_size();
    /**
     * Create buffers
     */
    this->buf_task_data_transform = new Buffer(*this->ctx,
                                    CL_MEM_READ_ONLY,
                                    sizeof(task_data_transform_t),
                                    (void *)NULL,
                                    &status);
    CHECK_CL_ERROR(status, "cl::Buffer");

    this->buf_taskdata = new Buffer(*this->ctx,
                                    CL_MEM_READ_ONLY ,
                                    task_data_buffer_size,
                                    NULL,
                                    &status);
    CHECK_CL_ERROR(status, "cl::Buffer");

    /**
     * Transfer to device
     */
    status = this->queue->enqueueWriteBuffer(*this->buf_task_data_transform,
                                CL_FALSE,
                                0,
                                sizeof(task_data_transform_t),
                                task_data_transform,
                                NULL,
                                NULL);
    CHECK_CL_ERROR(status, "cl::Queue.enqueueWriteBuffer()");

    status = this->queue->enqueueWriteBuffer(*this->buf_taskdata,
                                            CL_FALSE,
                                            0,
                                            task_data_buffer_size,
                                            task_data_buffer,
                                            NULL,
                                            NULL);
    CHECK_CL_ERROR(status, "cl::Queue.enqueueWriteBuffer()");
}

/**
 * Returns if all networks reached maxEpoch criteria.
 */
bool NetworkRunner::has_all_finished(neural_network_transform_t * transforms, int number_of_networks) {
    for (int i = 0; i < number_of_networks; i++) {
        if (transforms[i].state_epoch < transforms[i].criteria_maxEpochs) {
            // std::cout << transforms[i].state_epoch <<" "<<transforms[i].criteria_maxEpochs << std::endl << std::flush;
            return false;
        }
    }
    return true;
}

/**
 * Runs neural networks from container on GPU device.
 */
void NetworkRunner::run_networks(bool verbose) {
    cl_int status;
    this->networks_container->init_networks();

    void *neural_network_buffer = this->networks_container->get_neural_network_buffer();
    int neural_network_buffer_size = this->networks_container->get_neural_network_buffer_size();
    neural_network_transform_t * transforms = this->networks_container->get_transforms();
    int transforms_size = this->networks_container->get_transforms_size();
    int number_of_networks = this->networks_container->get_number_of_neural_networks();
    if (number_of_networks == 0) {
        return;
    }

    /**
     * Create buffers
     */
    Buffer buf_neural_network_transform(*this->ctx,
                    CL_MEM_READ_ONLY,
                    transforms_size,
                    (void *)NULL,
                    &status);
    CHECK_CL_ERROR(status, "cl::Buffer");

    Buffer buf_neuralnetwork(*this->ctx,
                    CL_MEM_READ_WRITE,
                    neural_network_buffer_size,
                    NULL,
                    &status);
    CHECK_CL_ERROR(status, "cl::Buffer");

    /**
     * Transfer to device
     */
    status = this->queue->enqueueWriteBuffer(   buf_neural_network_transform,
                                CL_FALSE,
                                0,
                                transforms_size,
                                transforms,
                                NULL,
                                NULL);
    CHECK_CL_ERROR(status, "cl::Queue.enqueueWriteBuffer()");
    
    status = this->queue->enqueueWriteBuffer(   buf_neuralnetwork,
                                CL_FALSE,
                                0,
                                neural_network_buffer_size,
                                neural_network_buffer,
                                NULL,
                                NULL);
    CHECK_CL_ERROR(status, "cl::Queue.enqueueWriteBuffer()");
   
    status = this->queue->finish();

    /**
     * Run kernel
     */
    this->knl_learn->setArg(0, buf_neural_network_transform);
    this->knl_learn->setArg(1, buf_neuralnetwork);
    this->knl_learn->setArg(2, *this->buf_task_data_transform);
    this->knl_learn->setArg(3, *this->buf_taskdata);
    this->knl_learn->setArg(4, number_of_networks);
    // return 0;
    // int global_x = 1024;
    // if (number_of_neurons * number_of_networks <= 1024) {
    //     global_x = number_of_neurons * number_of_networks;
    // }
    int dec_number_of_networks = number_of_networks - 1;
    int global_y = 1;
    while(dec_number_of_networks > 0) {
        dec_number_of_networks = dec_number_of_networks >> 1;
        global_y = global_y << 1;
    }
    int global_z = (global_y - 1) / 256 + 1;
    global_y = (global_y - 1) % 256 + 1;
    int rounded_neurons = this->max_neurons;
    for (int i = 1; i <= 1024; i *= 2) {
        if (rounded_neurons <= i) {
            rounded_neurons = i;
            break;
        }
    }
    if (verbose) {
        std::cout << " STARTING TRAINING:" << std::endl;
        // std::cout << " Workgroups " << rounded_neurons << "x" << global_y << "x" << global_z << std::endl;
    }
#ifdef USE_CHRONO
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#endif
    while(!this->has_all_finished(transforms, number_of_networks)) {
        status = this->queue->enqueueNDRangeKernel(*this->knl_learn,
                                                NullRange,
                                                // NDRange(global_x, global_y),
                                                // NDRange(number_of_neurons),
                                                NDRange(rounded_neurons, global_y, global_z),
                                                NDRange(rounded_neurons, 1, 1),
                                            NULL,
                                            NULL);
        CHECK_CL_ERROR(status, "cl::Queue.enqueueNDRangeKernel()");
        status = this->queue->finish();
        CHECK_CL_ERROR(status, "cl::Queue.finish()");

        status = this->queue->enqueueReadBuffer(   buf_neural_network_transform,
                                            CL_TRUE,
                                            0,
                                            transforms_size,
                                            transforms);
        CHECK_CL_ERROR(status, "cl::Queue.enqueueReadBuffer()");
        // break;
    }
    if (verbose) {
#ifdef USE_CHRONO
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::cout << "  Duration: " << (std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count()) / 1000000. << "s" << std::endl;
#endif    
    }
    status = this->queue->enqueueReadBuffer(   buf_neuralnetwork,
                                                CL_TRUE,
                                                0,
                                                neural_network_buffer_size,
                                                neural_network_buffer);
    CHECK_CL_ERROR(status, "cl::Queue.enqueueReadBuffer()"); 

    this->networks_container->update_networks();


    // return 0;
    // this->networks_container->neural_networks[0]->print(this->networks_container->taskData.learningOutputs);
    // this->networks_container->neural_networks[1]->print(this->networks_container->taskData.learningOutputs);
    // std::cout << "currentSquareErrorCounter " << this->networks_container->neural_networks[0]->currentSquareErrorCounter<<std::endl;
    if (verbose) {
        // std::cout << " accurancies: ";
        // for (int i = 0; i < this->networks_container->neural_networks[0]->criteria.maxEpochs; i++) {
        //     std::cout << this->networks_container->neural_networks[0]->squareErrorHistory[i] << " ";
        // }
        // std::cout << std::endl << std::endl;
    }
    // for (int i = 0; i < this->networks_container->neural_networks[0]->criteria.maxEpochs; i++) {
    //     std::cout << this->networks_container->neural_networks[1]->squareErrorHistory[i] << " ";
    // }
    // std::cout << std::endl;
    // for (int i = 0; i < this->networks_container->neural_networks[0]->criteria.maxEpochs; i++) {
    //     std::cout << this->networks_container->neural_networks[7]->squareErrorHistory[i] << " ";
    // }
    // std::cout<< std::endl << " / "<< this->networks_container->neural_networks[0]->state.learningLine<< std::endl;
    // std::cout << "lay "<< this->networks_container->neural_networks[0]->setup.numOfLayers << std::endl;
    // std::cout << ((float*)neural_network_buffer)[0] << std::endl;
    // std::cout << ((float*)neural_network_buffer)[1] <<  " " << transforms->neuralNetwork_b_size <<std::endl;
    // std::cout << ((float*)neural_network_buffer)[2] << " " << neural_network_buffer_size / 4 << std::endl;
    // std::cout << ((float*)neural_network_buffer)[5] << std::endl;
    // std::cout << ((float*)neural_network_buffer)[255] << std::endl << std::flush;
}

/**
 * Runs neural networks from container on GPU device.
 */
void NetworkRunner::run_networks_classification(int number_of_networks, bool verbose) {
    cl_int status;
    // this->networks_container->init_networks();
    void *neural_network_buffer = this->networks_container->get_neural_network_buffer();
    int neural_network_buffer_size = this->networks_container->get_neural_network_buffer_size();
    neural_network_transform_t * transforms = this->networks_container->get_transforms();
    int transforms_size = this->networks_container->get_transforms_size();
    if (number_of_networks == 0) {
        return;
    }

    /**
     * Create buffers
     */
    Buffer buf_neural_network_transform(*this->ctx,
                    CL_MEM_READ_ONLY,
                    transforms_size,
                    (void *)NULL,
                    &status);
    CHECK_CL_ERROR(status, "cl::Buffer");

    Buffer buf_neuralnetwork(*this->ctx,
                    CL_MEM_READ_WRITE,
                    neural_network_buffer_size,
                    NULL,
                    &status);
    CHECK_CL_ERROR(status, "cl::Buffer");

    /**
     * Transfer to device
     */
    status = this->queue->enqueueWriteBuffer(   buf_neural_network_transform,
                                CL_FALSE,
                                0,
                                transforms_size,
                                transforms,
                                NULL,
                                NULL);
    CHECK_CL_ERROR(status, "cl::Queue.enqueueWriteBuffer()");
    
    status = this->queue->enqueueWriteBuffer(   buf_neuralnetwork,
                                CL_FALSE,
                                0,
                                neural_network_buffer_size,
                                neural_network_buffer,
                                NULL,
                                NULL);
    CHECK_CL_ERROR(status, "cl::Queue.enqueueWriteBuffer()");
   
    status = this->queue->finish();
    // this->networks_container->neural_networks[0]->print(this->networks_container->taskData.learningOutputs);
    /**
     * Run kernel
     */
    this->knl_classification->setArg(0, buf_neural_network_transform);
    this->knl_classification->setArg(1, buf_neuralnetwork);
    this->knl_classification->setArg(2, *this->buf_task_data_transform);
    this->knl_classification->setArg(3, *this->buf_taskdata);
    this->knl_classification->setArg(4, number_of_networks);


    int dec_number_of_networks = number_of_networks - 1;
    int global_y = 1;
    while(dec_number_of_networks > 0) {
        dec_number_of_networks = dec_number_of_networks >> 1;
        global_y = global_y << 1;
    }
    int global_z = (global_y - 1) / 256 + 1;
    global_y = (global_y - 1) % 256 + 1;
    int rounded_neurons = this->max_neurons;
    for (int i = 1; i <= 1024; i *= 2) {
        if (rounded_neurons <= i) {
            rounded_neurons = i;
            break;
        }
    }
    if (verbose) {
        std::cout << " STARTING PREDICTION:" << std::endl;
        // std::cout << " Workgroups " << rounded_neurons << "x" << global_y << "x" << global_z << std::endl;
    }
#ifdef USE_CHRONO
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
#endif
    status = this->queue->enqueueNDRangeKernel(*this->knl_classification,
                                            NullRange,
                                            // NDRange(global_x, global_y),
                                            // NDRange(number_of_neurons),
                                            NDRange(rounded_neurons, global_y, global_z),
                                            NDRange(rounded_neurons, 1, 1),
                                        NULL,
                                        NULL);
    CHECK_CL_ERROR(status, "cl::Queue.enqueueNDRangeKernel()");
    status = this->queue->finish();
    CHECK_CL_ERROR(status, "cl::Queue.finish()");

    status = this->queue->enqueueReadBuffer(   buf_neural_network_transform,
                                        CL_TRUE,
                                        0,
                                        transforms_size,
                                        transforms);
    CHECK_CL_ERROR(status, "cl::Queue.enqueueReadBuffer()");

    if (verbose) {
#ifdef USE_CHRONO        
        std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        std::cout << "  Duration: " << (std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count()) / 1000000. << "s" << std::endl;
#endif
    }
    status = this->queue->enqueueReadBuffer(   buf_neuralnetwork,
                                                CL_TRUE,
                                                0,
                                                neural_network_buffer_size,
                                                neural_network_buffer);
    CHECK_CL_ERROR(status, "cl::Queue.enqueueReadBuffer()");

     status = this->queue->enqueueReadBuffer(*this->buf_taskdata,
                                            CL_TRUE,
                                            0,
                                            this->task_data_buffer_size,
                                            this->task_data_buffer);

    CHECK_CL_ERROR(status, "cl::Queue.enqueueReadBuffer()");

    // std::cout << ((float*)this->task_data_buffer)[6290] << " " << ((float*)this->task_data_buffer)[6291] << std::endl;

    this->networks_container->update_networks();
}