import { Product, Bill, Customer, Appointment } from './types';

const STORAGE_KEYS = {
  PRODUCTS: 'medical_store_products',
  BILLS: 'medical_store_bills',
  CUSTOMERS: 'medical_store_customers',
  APPOINTMENTS: 'medical_store_appointments',
};

// Initialize with sample data
export const initializeStorage = () => {
  if (!localStorage.getItem(STORAGE_KEYS.PRODUCTS)) {
    const sampleProducts: Product[] = [
      {
        id: '1',
        name: 'Ashwagandha Capsules',
        sku: 'ASH-001',
        category: 'Supplements',
        cost: 150,
        sellingPrice: 250,
        stock: 50,
        expiryDate: '2025-12-31',
      },
      {
        id: '2',
        name: 'Triphala Powder',
        sku: 'TRI-001',
        category: 'Powder',
        cost: 200,
        sellingPrice: 350,
        stock: 30,
        expiryDate: '2025-12-31',
      },
      {
        id: '3',
        name: 'Neem Oil',
        sku: 'NEEM-001',
        category: 'Oils',
        cost: 300,
        sellingPrice: 500,
        stock: 20,
        expiryDate: '2026-06-30',
      },
      {
        id: '4',
        name: 'Turmeric Capsules',
        sku: 'TUM-001',
        category: 'Supplements',
        cost: 100,
        sellingPrice: 180,
        stock: 75,
        expiryDate: '2025-12-31',
      },
      {
        id: '5',
        name: 'Brahmi Oil',
        sku: 'BRAH-001',
        category: 'Oils',
        cost: 250,
        sellingPrice: 450,
        stock: 15,
        expiryDate: '2025-11-30',
      },
    ];
    localStorage.setItem(STORAGE_KEYS.PRODUCTS, JSON.stringify(sampleProducts));
  }

  if (!localStorage.getItem(STORAGE_KEYS.CUSTOMERS)) {
    const sampleCustomers: Customer[] = [
      {
        id: '1',
        name: 'Rajesh Kumar',
        phone: '9876543210',
        email: 'rajesh@example.com',
        address: 'Delhi',
        registeredDate: '2024-01-15',
        totalPurchases: 5,
      },
    ];
    localStorage.setItem(STORAGE_KEYS.CUSTOMERS, JSON.stringify(sampleCustomers));
  }

  if (!localStorage.getItem(STORAGE_KEYS.BILLS)) {
    localStorage.setItem(STORAGE_KEYS.BILLS, JSON.stringify([]));
  }

  if (!localStorage.getItem(STORAGE_KEYS.APPOINTMENTS)) {
    localStorage.setItem(STORAGE_KEYS.APPOINTMENTS, JSON.stringify([]));
  }
};

// Products
export const getProducts = (): Product[] => {
  return JSON.parse(localStorage.getItem(STORAGE_KEYS.PRODUCTS) || '[]');
};

export const saveProduct = (product: Product) => {
  const products = getProducts();
  const index = products.findIndex(p => p.id === product.id);
  if (index >= 0) {
    products[index] = product;
  } else {
    products.push(product);
  }
  localStorage.setItem(STORAGE_KEYS.PRODUCTS, JSON.stringify(products));
};

export const deleteProduct = (id: string) => {
  const products = getProducts().filter(p => p.id !== id);
  localStorage.setItem(STORAGE_KEYS.PRODUCTS, JSON.stringify(products));
};

// Bills
export const getBills = (): Bill[] => {
  return JSON.parse(localStorage.getItem(STORAGE_KEYS.BILLS) || '[]');
};

export const saveBill = (bill: Bill) => {
  const bills = getBills();
  bills.push(bill);
  localStorage.setItem(STORAGE_KEYS.BILLS, JSON.stringify(bills));
};

// Customers
export const getCustomers = (): Customer[] => {
  return JSON.parse(localStorage.getItem(STORAGE_KEYS.CUSTOMERS) || '[]');
};

export const saveCustomer = (customer: Customer) => {
  const customers = getCustomers();
  const index = customers.findIndex(c => c.id === customer.id);
  if (index >= 0) {
    customers[index] = customer;
  } else {
    customers.push(customer);
  }
  localStorage.setItem(STORAGE_KEYS.CUSTOMERS, JSON.stringify(customers));
};

// Appointments
export const getAppointments = (): Appointment[] => {
  return JSON.parse(localStorage.getItem(STORAGE_KEYS.APPOINTMENTS) || '[]');
};

export const saveAppointment = (appointment: Appointment) => {
  const appointments = getAppointments();
  appointments.push(appointment);
  localStorage.setItem(STORAGE_KEYS.APPOINTMENTS, JSON.stringify(appointments));
};

export const updateAppointment = (id: string, status: 'scheduled' | 'completed' | 'cancelled') => {
  const appointments = getAppointments();
  const index = appointments.findIndex(a => a.id === id);
  if (index >= 0) {
    appointments[index].status = status;
    localStorage.setItem(STORAGE_KEYS.APPOINTMENTS, JSON.stringify(appointments));
  }
};
