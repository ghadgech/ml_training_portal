export interface Product {
  id: string;
  name: string;
  sku: string;
  category: string;
  cost: number;
  sellingPrice: number;
  stock: number;
  expiryDate?: string;
}

export interface CartItem {
  productId: string;
  quantity: number;
  price: number;
}

export interface Bill {
  id: string;
  date: string;
  items: CartItem[];
  customerId?: string;
  totalAmount: number;
  paymentMethod: 'cash' | 'card' | 'upi';
  notes?: string;
}

export interface Customer {
  id: string;
  name: string;
  phone: string;
  email?: string;
  address?: string;
  registeredDate: string;
  totalPurchases: number;
}

export interface Appointment {
  id: string;
  customerName: string;
  customerPhone: string;
  date: string;
  time: string;
  doctorName: string;
  notes?: string;
  status: 'scheduled' | 'completed' | 'cancelled';
}
