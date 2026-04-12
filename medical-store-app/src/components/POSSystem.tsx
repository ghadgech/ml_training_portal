import React, { useState } from 'react';
import { Trash2, Plus, X } from 'lucide-react';
import { Product, CartItem, Bill } from '../types';
import { getProducts, getCustomers, saveBill } from '../storage';
import '../styles/POSSystem.css';

export default function POSSystem() {
  const [cart, setCart] = useState<CartItem[]>([]);
  const [selectedCustomer, setSelectedCustomer] = useState<string>('');
  const [paymentMethod, setPaymentMethod] = useState<'cash' | 'card' | 'upi'>('cash');
  const [showBillPreview, setShowBillPreview] = useState(false);
  const [notes, setNotes] = useState('');

  const products = getProducts();
  const customers = getCustomers();

  const addToCart = (product: Product) => {
    if (product.stock === 0) {
      alert('Out of stock!');
      return;
    }

    const existingItem = cart.find(item => item.productId === product.id);
    if (existingItem) {
      if (existingItem.quantity < product.stock) {
        setCart(cart.map(item =>
          item.productId === product.id
            ? { ...item, quantity: item.quantity + 1 }
            : item
        ));
      } else {
        alert('Insufficient stock');
      }
    } else {
      setCart([...cart, { productId: product.id, quantity: 1, price: product.sellingPrice }]);
    }
  };

  const updateQuantity = (productId: string, quantity: number) => {
    const product = products.find(p => p.id === productId);
    if (product && quantity > product.stock) {
      alert(`Only ${product.stock} items available`);
      return;
    }

    if (quantity <= 0) {
      removeFromCart(productId);
    } else {
      setCart(cart.map(item =>
        item.productId === productId ? { ...item, quantity } : item
      ));
    }
  };

  const removeFromCart = (productId: string) => {
    setCart(cart.filter(item => item.productId !== productId));
  };

  const calculateTotal = () => {
    return cart.reduce((total, item) => total + item.price * item.quantity, 0);
  };

  const calculateProfit = () => {
    return cart.reduce((profit, item) => {
      const product = products.find(p => p.id === item.productId);
      if (product) {
        profit += (product.sellingPrice - product.cost) * item.quantity;
      }
      return profit;
    }, 0);
  };

  const completeBill = () => {
    if (cart.length === 0) {
      alert('Cart is empty!');
      return;
    }

    const bill: Bill = {
      id: `BILL-${Date.now()}`,
      date: new Date().toISOString(),
      items: cart,
      customerId: selectedCustomer || undefined,
      totalAmount: calculateTotal(),
      paymentMethod,
      notes,
    };

    saveBill(bill);
    alert(`Bill #${bill.id} completed!`);
    setCart([]);
    setSelectedCustomer('');
    setNotes('');
    setPaymentMethod('cash');
  };

  const getProductName = (productId: string) => {
    return products.find(p => p.id === productId)?.name || 'Unknown';
  };

  const total = calculateTotal();
  const profit = calculateProfit();

  return (
    <div className="pos-system">
      <div className="pos-grid">
        {/* Products Section */}
        <div className="pos-section products-section">
          <h2 className="card-title">Products</h2>
          <div className="products-grid">
            {products.map(product => (
              <div
                key={product.id}
                className={`product-card ${product.stock === 0 ? 'out-of-stock' : ''}`}
              >
                <div className="product-info">
                  <h4>{product.name}</h4>
                  <p className="product-sku">{product.sku}</p>
                  <p className="product-price">₹{product.sellingPrice}</p>
                  <p className="product-stock">Stock: {product.stock}</p>
                </div>
                <button
                  className="btn-primary"
                  onClick={() => addToCart(product)}
                  disabled={product.stock === 0}
                >
                  <Plus size={18} /> Add
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* Cart Section */}
        <div className="pos-section cart-section">
          <h2 className="card-title">Shopping Cart</h2>

          {cart.length === 0 ? (
            <p className="empty-cart">Cart is empty</p>
          ) : (
            <>
              <div className="cart-items">
                {cart.map(item => {
                  const product = products.find(p => p.id === item.productId);
                  return (
                    <div key={item.productId} className="cart-item">
                      <div className="cart-item-info">
                        <h4>{getProductName(item.productId)}</h4>
                        <p className="cart-item-price">₹{item.price}</p>
                      </div>
                      <div className="cart-item-qty">
                        <button
                          className="qty-btn"
                          onClick={() => updateQuantity(item.productId, item.quantity - 1)}
                        >
                          -
                        </button>
                        <input
                          type="number"
                          min="1"
                          value={item.quantity}
                          onChange={e =>
                            updateQuantity(item.productId, parseInt(e.target.value) || 1)
                          }
                          className="qty-input"
                        />
                        <button
                          className="qty-btn"
                          onClick={() => updateQuantity(item.productId, item.quantity + 1)}
                        >
                          +
                        </button>
                      </div>
                      <div className="cart-item-total">
                        ₹{item.price * item.quantity}
                      </div>
                      <button
                        className="btn-delete"
                        onClick={() => removeFromCart(item.productId)}
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  );
                })}
              </div>

              <div className="cart-summary">
                <div className="summary-row">
                  <span>Subtotal:</span>
                  <strong>₹{total.toFixed(2)}</strong>
                </div>
                <div className="summary-row">
                  <span>Profit:</span>
                  <strong className="profit">₹{profit.toFixed(2)}</strong>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Customer (Optional)</label>
                <select
                  className="form-select"
                  value={selectedCustomer}
                  onChange={e => setSelectedCustomer(e.target.value)}
                >
                  <option value="">Walk-in Customer</option>
                  {customers.map(customer => (
                    <option key={customer.id} value={customer.id}>
                      {customer.name} - {customer.phone}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Payment Method</label>
                <select
                  className="form-select"
                  value={paymentMethod}
                  onChange={e => setPaymentMethod(e.target.value as any)}
                >
                  <option value="cash">Cash</option>
                  <option value="card">Card</option>
                  <option value="upi">UPI</option>
                </select>
              </div>

              <div className="form-group">
                <label className="form-label">Notes</label>
                <textarea
                  className="form-textarea"
                  value={notes}
                  onChange={e => setNotes(e.target.value)}
                  placeholder="Add notes if any..."
                  rows={3}
                />
              </div>

              <button className="btn-success button" onClick={completeBill}>
                Complete Bill (₹{total.toFixed(2)})
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
