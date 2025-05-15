#ifndef DEVICES_CUH
#define DEVICES_CUH

#include <string>
#include <vector>
#include <unordered_map>
#include <memory> // For std::unique_ptr

class Devices {
public:
    Devices();
    ~Devices(); // Necesario para PIMPL

    std::unordered_map<std::string, std::string> get_properties(int device_id) const;
    void print_devices() const;
    std::vector<std::unordered_map<std::string, std::string>> get_devices() const;

private:
    class Impl; // Declaración opaca
    std::unique_ptr<Impl> impl_; // PIMPL
};

#endif